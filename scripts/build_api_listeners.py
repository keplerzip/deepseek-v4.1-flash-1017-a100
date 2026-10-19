"""Build a fail-closed, IPv4-only API policy from Docker network inspect JSON.

Run inside the pinned image with --network host. No Docker SDK or extra tools.
"""
import argparse
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import struct
import tempfile


def host_interfaces():
    """Read actual interfaces/IPv4 addresses from Linux rtnetlink, not sysfs names."""
    result = {}
    for index, name in socket.if_nameindex():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            flags = struct.unpack_from('H', fcntl.ioctl(
                probe.fileno(), 0x8913, struct.pack('256s', name.encode()[:15])), 16)[0]
        result[index] = {'name': name, 'up': bool(flags & 1),
                         'bridge': (Path('/sys/class/net') / name / 'bridge').is_dir(),
                         'addresses': []}
    with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_ROUTE) as route:
        route.settimeout(5)
        route.bind((0, 0))
        # RTM_GETADDR | NLM_F_REQUEST | NLM_F_DUMP, ifaddrmsg(AF_INET).
        route.send(struct.pack('IHHII', 24, 22, 0x301, 1, 0) +
                   struct.pack('BBBBI', socket.AF_INET, 0, 0, 0, 0))
        done = False
        while not done:
            data = route.recv(1024 * 1024)
            offset = 0
            while offset + 16 <= len(data):
                length, kind, _, seq, _ = struct.unpack_from('IHHII', data, offset)
                if length < 16 or offset + length > len(data):
                    raise ValueError('Malformed address dump')
                payload = data[offset + 16:offset + length]
                offset += (length + 3) & ~3
                if seq != 1:
                    continue
                if kind == 3:
                    done = True
                    break
                if kind == 2:
                    raise OSError('Address inventory failed')
                if kind != 20 or len(payload) < 8:
                    continue
                family, prefix, _, _, index = struct.unpack_from('BBBBI', payload)
                if family != socket.AF_INET or index not in result:
                    continue
                attrs, cursor = {}, 8
                while cursor + 4 <= len(payload):
                    size, attr_type = struct.unpack_from('HH', payload, cursor)
                    if size < 4 or cursor + size > len(payload):
                        raise ValueError('Malformed address attribute')
                    attrs[attr_type] = payload[cursor + 4:cursor + size]
                    cursor += (size + 3) & ~3
                raw = attrs.get(2, attrs.get(1))  # IFA_LOCAL before IFA_ADDRESS.
                if raw is not None and len(raw) == 4:
                    result[index]['addresses'].append(f'{socket.inet_ntop(socket.AF_INET, raw)}/{prefix}')
    return list(result.values())


def compile_policy(networks, interfaces, port=8005):
    if port != 8005:
        raise ValueError('This deployment fixes the API port at 8005')
    if not isinstance(networks, list):
        raise ValueError('docker network inspect must return a JSON list')
    actual = {item['name']: item for item in interfaces}
    gateways, pending, skipped, verified_sources = {}, [], [], []
    active_subnets = set()
    non_bridge_networks = []
    for interface in interfaces:
        if interface['bridge'] or interface['name'] == 'lo':
            continue
        for address in interface['addresses']:
            non_bridge_networks.append(ipaddress.ip_interface(address).network)
    for network in networks:
        if network.get('Driver') != 'bridge' or network.get('Ingress'):
            skipped.append({'network': network.get('Name'), 'reason': 'not a non-ingress Docker bridge'})
            continue
        options = network.get('Options') or {}
        name, identity = network.get('Name'), network.get('Id', '')
        interface_name = options.get('com.docker.network.bridge.name')
        if not interface_name:
            if name == 'bridge' or options.get('com.docker.network.bridge.default_bridge') == 'true':
                interface_name = 'docker0'
            elif re.fullmatch('[0-9a-f]{64}', identity):
                interface_name = 'br-' + identity[:12]
            else:
                raise ValueError('Docker bridge has no valid interface identity')
        if not isinstance(interface_name, str) or not re.fullmatch('[A-Za-z0-9_.-]{1,15}', interface_name):
            raise ValueError('Invalid Docker bridge interface name')
        for entry in (network.get('IPAM') or {}).get('Config') or []:
            subnet_text = entry.get('Subnet')
            if not subnet_text:
                skipped.append({'network': name, 'reason': 'no IPAM subnet'})
                continue
            subnet = ipaddress.ip_network(subnet_text, strict=True)
            if subnet.version != 4:
                skipped.append({'network': name, 'subnet': str(subnet), 'reason': 'IPv6 explicitly unsupported'})
                continue
            forbidden = ('0.0.0.0/8', '127.0.0.0/8', '169.254.0.0/16', '224.0.0.0/4', '240.0.0.0/4')
            if subnet.prefixlen < 8 or any(subnet.overlaps(ipaddress.ip_network(cidr)) for cidr in forbidden):
                raise ValueError(f'Unsafe Docker IPv4 subnet: {subnet}')
            gateway_text = entry.get('Gateway')
            if not gateway_text:
                pending.append({'network': name, 'subnet': str(subnet), 'reason': 'no IPv4 gateway'})
                continue
            gateway = ipaddress.ip_address(gateway_text)
            if gateway.version != 4 or gateway not in subnet or gateway in (subnet.network_address, subnet.broadcast_address):
                raise ValueError(f'Invalid gateway for Docker subnet: {gateway_text}')
            if any(subnet.overlaps(other) for other in non_bridge_networks):
                raise ValueError(f'Docker subnet overlaps a non-bridge host interface: {subnet}')
            interface = actual.get(interface_name)
            assigned = set(interface['addresses']) if interface else set()
            if not interface or not interface['up'] or not interface['bridge'] or f'{gateway}/{subnet.prefixlen}' not in assigned:
                pending.append({'network': name, 'interface': interface_name, 'gateway': str(gateway),
                                'subnet': str(subnet), 'reason': 'gateway not active on the expected host bridge'})
                continue
            active_subnets.add(str(subnet))
            verified_sources.append({'network': name, 'interface': interface_name,
                                     'gateway': str(gateway), 'subnet': str(subnet)})
            is_default = name == 'bridge' or options.get('com.docker.network.bridge.default_bridge') == 'true'
            if not is_default:
                continue
            if str(gateway) in gateways and gateways[str(gateway)]['interface'] != interface_name:
                raise ValueError('One gateway resolves to multiple Docker interfaces')
            gateways[str(gateway)] = {'network': name, 'interface': interface_name, 'gateway': str(gateway),
                                      'default_bridge': True,
                                      'subnet': str(subnet)}
    # R3.9 host.docker.internal uses docker0 even for a custom bridge client.
    # Permit every verified Docker source subnet at every verified gateway.
    allowed = sorted(active_subnets)
    listeners = [{'address': '127.0.0.1', 'port': port, 'kind': 'loopback', 'allowed_sources': ['127.0.0.1/32']}]
    if len(gateways) > 1:
        raise ValueError('Default Docker bridge must have one IPv4 API gateway')
    for gateway, metadata in sorted(gateways.items()):
        listeners.append({'address': gateway, 'port': port, 'kind': 'docker-bridge',
                          'allowed_sources': allowed, **metadata})
    return {'schema': 1, 'port': port, 'authentication': 'disabled',
            'source_identity': 'TCP peer address; HTTP forwarding headers are ignored',
            'scope': 'localhost and all active Docker IPv4 bridge networks',
            'refresh': 'Restart this deployment after adding or changing Docker networks',
            'listeners': listeners, 'docker_sources': verified_sources,
            'pending': pending, 'skipped': skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--networks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8005)
    args = parser.parse_args()
    policy = compile_policy(json.loads(args.networks.read_text()), host_interfaces(), args.port)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=args.output.parent, prefix=args.output.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(policy, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, args.output)
    print(json.dumps(policy, ensure_ascii=False))


if __name__ == '__main__':
    main()
