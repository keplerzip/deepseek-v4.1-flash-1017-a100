"""CPU policy checks and real localhost TCP/UDS forwarding; no GPU claims."""
import asyncio
import copy
import importlib.util
import ipaddress
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load('build_api_listeners')
frontend = load('frontend')


def network(name='bridge', identity='a' * 64, subnet='172.17.0.0/16', gateway='172.17.0.1', driver='bridge'):
    return {'Name': name, 'Id': identity, 'Driver': driver, 'Ingress': False,
            'Options': {}, 'IPAM': {'Config': [{'Subnet': subnet, 'Gateway': gateway}]}}


def interfaces():
    return [{'name': 'lo', 'up': True, 'bridge': False, 'addresses': ['127.0.0.1/8']},
            {'name': 'eth0', 'up': True, 'bridge': False, 'addresses': ['192.0.2.10/24']},
            {'name': 'docker0', 'up': True, 'bridge': True, 'addresses': ['172.17.0.1/16']},
            {'name': 'br-bbbbbbbbbbbb', 'up': True, 'bridge': True, 'addresses': ['172.18.0.1/16']}]


class PolicyTests(unittest.TestCase):
    def make_policy(self):
        return builder.compile_policy([network(), network('app', 'b' * 64, '172.18.0.0/16', '172.18.0.1')], interfaces())

    def parse(self, policy):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'policy.json'
            path.write_text(json.dumps(policy))
            return frontend.load_policy(path)

    def test_r39_exact_entries_all_bridge_sources(self):
        policy = self.make_policy()
        self.assertEqual([row['address'] for row in policy['listeners']], ['127.0.0.1', '172.17.0.1'])
        self.assertEqual(policy['listeners'][1]['allowed_sources'], ['172.17.0.0/16', '172.18.0.0/16'])
        rows = self.parse(policy)
        self.assertTrue(rows[1].permits('172.18.0.50'))
        self.assertFalse(rows[1].permits('192.0.2.10'))
        self.assertFalse(rows[1].permits('8.8.8.8'))
        self.assertFalse(rows[1].permits('::1'))

    def test_no_wildcard_or_custom_gateway_bind(self):
        policy = self.make_policy()
        self.assertNotIn('0.0.0.0', [row['address'] for row in policy['listeners']])
        self.assertNotIn('172.18.0.1', [row['address'] for row in policy['listeners']])
        policy['listeners'][1]['address'] = '0.0.0.0'
        with self.assertRaises(ValueError):
            self.parse(policy)

    def test_ipv6_explicit_skip(self):
        data = network()
        data['IPAM']['Config'].append({'Subnet': 'fd00:123::/64', 'Gateway': 'fd00:123::1'})
        policy = builder.compile_policy([data], interfaces())
        self.assertEqual(policy['skipped'][0]['reason'], 'IPv6 explicitly unsupported')
        self.assertEqual(len(policy['listeners']), 2)

    def test_invalid_cidr_and_gateway_rejected(self):
        for subnet, gateway in [('172.17.0.0/99', '172.17.0.1'), ('172.17.0.0/16', '192.0.2.1'),
                                ('0.0.0.0/0', '172.17.0.1'), ('127.0.0.0/8', '127.0.0.2')]:
            with self.subTest(subnet=subnet, gateway=gateway), self.assertRaises(ValueError):
                builder.compile_policy([network(subnet=subnet, gateway=gateway)], interfaces())

    def test_host_lan_gateway_is_rejected(self):
        data = network(subnet='192.0.2.0/24', gateway='192.0.2.10')
        data['Options']['com.docker.network.bridge.name'] = 'eth0'
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            builder.compile_policy([data], interfaces())

    def test_missing_inactive_bridge_is_pending(self):
        inventory = interfaces()
        inventory[2]['up'] = False
        policy = builder.compile_policy([network()], inventory)
        self.assertEqual([row['address'] for row in policy['listeners']], ['127.0.0.1'])
        self.assertEqual(len(policy['pending']), 1)
        self.assertFalse(policy['docker_sources'])

    def test_nonbridge_and_ingress_not_allowed(self):
        data = network()
        data['Ingress'] = True
        policy = builder.compile_policy([data, network(driver='overlay')], interfaces())
        self.assertEqual(len(policy['skipped']), 2)
        self.assertEqual(len(policy['listeners']), 1)

    def test_fixed_port_and_bad_policy_fail_closed(self):
        with self.assertRaises(ValueError):
            builder.compile_policy([], interfaces(), 8007)
        policy = self.make_policy()
        policy['listeners'][1]['allowed_sources'].append('192.0.2.0/24')
        with self.assertRaises(ValueError):
            self.parse(policy)

    def test_real_inventory_uses_real_interfaces(self):
        inventory = builder.host_interfaces()
        self.assertEqual({row['name'] for row in inventory}, {name for _, name in socket.if_nameindex()})
        self.assertIn('127.0.0.1/8', next(row for row in inventory if row['name'] == 'lo')['addresses'])


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'engine.sock'
        self.received = []
        self.response_chunks = [b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n',
                                b'data: {"token":"Chinese: ', '中文'.encode(), b'"}\n\n', b'data: [DONE]\n\n']

        async def upstream(reader, writer):
            try:
                self.received.append(await reader.readuntil(b'\r\n\r\n'))
                for chunk in self.response_chunks:
                    writer.write(chunk)
                    await writer.drain()
                    await asyncio.sleep(0.005)
            finally:
                await frontend.close_writer(writer)

        self.upstream = await asyncio.start_unix_server(upstream, str(self.path))
        self.servers = []

    async def asyncTearDown(self):
        for server in self.servers + [self.upstream]:
            server.close()
            await server.wait_closed()
        await asyncio.sleep(0)
        self.temporary.cleanup()

    async def start(self, address='127.0.0.1', allowed='127.0.0.1/32'):
        # Internal localhost-only fixture simulates a bridge bind; production
        # load_policy rejects loopback addresses for docker-bridge listeners.
        listener = frontend.Listener(address, 0, (ipaddress.ip_network(allowed),), 'loopback')
        servers = await frontend.bind_listeners([listener], self.path)
        self.servers.extend(servers)
        return servers[0].sockets[0].getsockname()

    async def request(self, endpoint, source='127.0.0.1', headers=b''):
        reader, writer = await asyncio.open_connection(*endpoint, local_addr=(source, 0))
        try:
            writer.write(b'GET /v1/models HTTP/1.1\r\nHost: localhost\r\n' + headers + b'\r\n')
            await writer.drain()
            response = bytearray()
            while True:
                try:
                    chunk = await asyncio.wait_for(reader.read(65536), 3)
                except ConnectionResetError:
                    break
                if not chunk:
                    break
                response.extend(chunk)
            return bytes(response)
        finally:
            await frontend.close_writer(writer)

    async def test_no_authorization_stream_is_byte_exact(self):
        endpoint = await self.start()
        response = await self.request(endpoint)
        self.assertEqual(response, b''.join(self.response_chunks))
        self.assertEqual(len(self.received), 1)
        self.assertNotIn(b'Authorization', self.received[0])

    async def test_unapproved_real_socket_source_is_denied(self):
        endpoint = await self.start()
        response = await self.request(endpoint, source='127.0.0.2')
        self.assertIn(b'403 Forbidden', response)
        self.assertFalse(self.received)

    async def test_x_forwarded_for_cannot_override_peer(self):
        endpoint = await self.start()
        response = await self.request(endpoint, source='127.0.0.2',
                                      headers=b'X-Forwarded-For: 127.0.0.1\r\nForwarded: for=127.0.0.1\r\n')
        self.assertIn(b'403 Forbidden', response)
        self.assertFalse(self.received)

    async def test_specific_gateway_fixture_no_wildcard_listener(self):
        endpoint = await self.start(address='127.0.0.2')
        self.assertEqual(endpoint[0], '127.0.0.2')
        response = await self.request(endpoint)
        self.assertEqual(response, b''.join(self.response_chunks))
        with self.assertRaises(OSError):
            reader, writer = await asyncio.open_connection('127.0.0.1', endpoint[1])
            writer.close()
            await writer.wait_closed()

    async def test_source_subnet_membership_uses_peer_not_headers(self):
        endpoint = await self.start(allowed='127.0.0.0/30')
        response = await self.request(endpoint, source='127.0.0.2', headers=b'X-Forwarded-For: 8.8.8.8\r\n')
        self.assertEqual(response, b''.join(self.response_chunks))
        self.assertIn(b'X-Forwarded-For: 8.8.8.8', self.received[0])

    async def test_request_body_and_large_stream_roundtrip(self):
        # Raw payload through UDS must retain binary data and backpressure.
        self.upstream.close()
        await self.upstream.wait_closed()
        self.path.unlink()
        payload = bytes(range(256)) * 8192

        async def echo(reader, writer):
            while chunk := await reader.read(16384):
                writer.write(chunk)
                await writer.drain()
            await frontend.close_writer(writer)

        self.upstream = await asyncio.start_unix_server(echo, str(self.path))
        endpoint = await self.start()
        reader, writer = await asyncio.open_connection(*endpoint)
        writer.write(payload)
        await writer.drain()
        writer.write_eof()
        self.assertEqual(await asyncio.wait_for(reader.readexactly(len(payload)), 5), payload)
        self.assertEqual(await asyncio.wait_for(reader.read(), 3), b'')
        await frontend.close_writer(writer)


if __name__ == '__main__':
    unittest.main(verbosity=2)
