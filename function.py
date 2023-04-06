import numpy as np
from scapy.all import *

def get_features_test(pcap, feature_path):
    pkts = rdpcap(pcap)
    for i, pkt in enumerate(pkts):
        print(pkt['TCP'].payload.build()[:2] )
    with open(feature_path, 'w') as file_object:
        file_object.write('123"')



def get_features(pcapfile,output_file_path,print_info=False):
    tls_type = {1:'Client Hello',2:'Server Hello',11:'Certificate',12:'Server Key Exchange',14:'Server Hello Done',
                16:'Client Key Exchange',4:'New Session Ticket'}
    error_info = []
    pkts = rdpcap(pcapfile)
    length_seq = list()
    interval_sqe = list()
    tls_bs = list()
    existing_pkt = list()
    flow_start = False

    for i, pkt in enumerate(pkts):
        if len(pkt['TCP'].payload.build()) > 9:
            if (pkt['TCP'].payload.build()[:2] == b'\x16\x03' \
                and pkt['TCP'].payload.build()[5] == 1 and pkt['TCP'].payload.build()[9] == 3) \
                    or (pkt['TCP'].payload.build()[2:5] in [b'\x01\x03\x01',b'\x01\x03\x00',b'\x01\x00\x02'] \
                        and pkt['TCP'].payload.build()[0] > 127):
                flow_start = True
                client_src = pkt.src
        if not flow_start:
            continue
        if [pkt['TCP'].seq,pkt['TCP'].ack] in existing_pkt:
            continue
        else:
            existing_pkt.append([pkt['TCP'].seq,pkt['TCP'].ack])
        if pkt.src == client_src:
            length_seq.append(-pkt.len)
            interval_sqe.append(-float(pkt.time - pkts[0].time))
        else:
            length_seq.append(pkt.len)
            interval_sqe.append(float(pkt.time - pkts[0].time))

        ip_tcp_hdr_overhead = ((pkt['IP'].ihl * 4) + (pkt['TCP'].dataofs * 4))
        tls_bs.append((pkt['TCP'].payload).build()[:pkt['IP'].len - ip_tcp_hdr_overhead])

    interval_sqe = interval_sqe[1:]
    np_length_seq = np.array(length_seq)
    np_interval_sqe = np.array(interval_sqe)
    tls_raw_bytes = b''.join(tls_bs)
    if print_info:
        print(length_seq)
        print(interval_sqe)

    # 得到两种序列，接下来提取tls层报文，得到每个tls报文
    tls_pkts = []
    flow_encrypted = False
    flow_intact = False
    flow_handshake_over = False

    while tls_raw_bytes and not error_info:
        if print_info:
            print(len(tls_raw_bytes))
            if len(tls_raw_bytes) > 80:
                print(tls_raw_bytes[:80])
            else:
                print(tls_raw_bytes)

        if tls_raw_bytes[0] > 127 and tls_raw_bytes[2] == 4:
            if print_info:
                print('SSLv2 的 Server Hello')
            break

        if flow_encrypted:
            tls_len = tls_raw_bytes[4] + tls_raw_bytes[3] * 256
            tls_raw_bytes = tls_raw_bytes[5 + tls_len:]
            flow_encrypted = False
            continue

        if tls_raw_bytes[0] == 23:
            flow_intact = True
            if print_info:
                print('传输数据 Application ，停止解析')
            break
        if len(tls_raw_bytes) > 4:
            tls_len = tls_raw_bytes[4] + tls_raw_bytes[3] * 256
        else:
            break

        if tls_raw_bytes[0] == 20:
            tls_raw_bytes = tls_raw_bytes[5 + tls_len:]
            flow_encrypted = True
            flow_handshake_over = True
            if print_info:
                print('Change Cipher Spec，之后信息加密')
            continue

        if (tls_raw_bytes[:2] == b'\x16\x03' and tls_raw_bytes[5] == 1 and tls_raw_bytes[9] == 3) \
                or (tls_raw_bytes[2:5] in [b'\x01\x03\x01',b'\x01\x03\x00',b'\x01\x00\x02'] and tls_raw_bytes[0] > 127):
            flow_encrypted = False

        if tls_raw_bytes[0] == 21:
            tls_raw_bytes = tls_raw_bytes[5 + tls_len:]
            if print_info:
                print('Alert')
            continue

        if tls_raw_bytes[0] > 127 and tls_raw_bytes[2:5] in [b'\x01\x03\x01',b'\x01\x03\x00',b'\x01\x00\x02']:
            tls_len = tls_raw_bytes[1] + tls_raw_bytes[0] * 256 - 32768
            p = tls_raw_bytes[:2 + tls_len]
            tls_pkts.append(p)
            tls_raw_bytes = tls_raw_bytes[2 + tls_len:]
            if print_info:
                print('SSLv2 的 Client Hello')
            continue
        ps = tls_raw_bytes[5:5 + tls_len]
        tls_raw_bytes = tls_raw_bytes[5 + tls_len:]
        while ps and not flow_encrypted and not flow_handshake_over:
            p_len = ps[3] + ps[2] * 256 + ps[1] * 256 * 256
            p = ps[:4 + p_len]
            ps = ps[4 + p_len:]
            if p[0] in tls_type.keys():
                tls_pkts.append(p)
                if print_info:
                    print('得到数据包 ' + tls_type[p[0]])
            else:
                error_info.append('Error getting TLS packets. \n')
                break

    ch,sh,cert = '','',''
    for p in tls_pkts:
        if (p[0] == 1 and p[4] == 3 ) or (p[2:5] in [b'\x01\x03\x01',b'\x01\x03\x00',b'\x01\x00\x02'] and p[0] > 127):
            ch = p
        elif p[0] == 2:
            sh = p
        elif p[0] == 11:
            cert = p

    if print_info:
        print('TLS info : ')
        print(ch)
        print(sh)
        print(cert)

    if not flow_intact:
        error_info.append('The flow does not transmited data. \n')
    if not sh:
        error_info.append('The flow had not Server Hello. \n')
    if error_info:
        print(''.join(error_info))
        return -1

    file_mode = 'w' if not os.path.exists(output_file_path) else 'a'
    with open(output_file_path,file_mode) as output_file:
        output_file.write('\"')
        for n,a in enumerate(length_seq):
            if n < 200:
                output_file.write(str(a) + ',')
        output_file.write('\",')
        output_file.write('\",')
        # feature 0
        for n, a in enumerate(interval_sqe):
            if n < 200:
                output_file.write(str(a) + ',')
        output_file.write('\",')
        # feature 1
        for p in [ch,sh,cert]:
            output_file.write('\"')
            if p:
                for s in p:
                    output_file.write(str(s))
                    output_file.write(',')
            output_file.write('\",')
        # feature 2,3,4

        features = list()
        features.append(np_length_seq.shape[0]) # 会话总包数5
        features.append(np_length_seq[np_length_seq < 0].shape[0]) # client包数6
        features.append(np_length_seq[np_length_seq > 0].shape[0]) # Server包数7
        features.append(np_length_seq[np_length_seq < 0].shape[0] / np_length_seq[np_length_seq > 0].shape[0]) # C/S包比率8

        features.append(abs(np_length_seq).sum()) # 会话总包长9
        features.append(-np_length_seq[np_length_seq < 0].sum()) # Client总包长10
        features.append(np_length_seq[np_length_seq > 0].sum()) # Server总包长11
        features.append(-np_length_seq[np_length_seq < 0].sum() / np_length_seq[np_length_seq > 0].sum()) # C/S总包长比率12

        features.append(abs(np_length_seq).max()) # 会话包长最大值、最小值、平均数、方差13 14 15 16
        features.append(abs(np_length_seq).min())
        features.append(abs(np_length_seq).mean())
        features.append(abs(np_length_seq).var())
        features.append((-np_length_seq[np_length_seq < 0]).max()) # client包长最大值、最小值、平均数、方差17 18 19 20
        features.append((-np_length_seq[np_length_seq < 0]).min())
        features.append((-np_length_seq[np_length_seq < 0]).mean())
        features.append((-np_length_seq[np_length_seq < 0]).var())
        features.append(np_length_seq[np_length_seq > 0].max()) # server包长最大值、最小值、平均数、方差21 22 23 24
        features.append(np_length_seq[np_length_seq > 0].min())
        features.append(np_length_seq[np_length_seq > 0].mean())
        features.append(np_length_seq[np_length_seq > 0].var())
        features.append((abs(np_interval_sqe) - np.append([0], abs(np_interval_sqe)[:-1])).max()) # 会话包到达时间最大值、最小值、平均数、方差25 26 27 28
        features.append((abs(np_interval_sqe) - np.append([0], abs(np_interval_sqe)[:-1])).min())
        features.append((abs(np_interval_sqe) - np.append([0], abs(np_interval_sqe)[:-1])).mean())
        features.append((abs(np_interval_sqe) - np.append([0], abs(np_interval_sqe)[:-1])).var())
        np_client_interval_sqe = (-np_interval_sqe[np_interval_sqe < 0]) \
                                 - np.append([0],(-np_interval_sqe[np_interval_sqe < 0])[:-1])
        np_server_interval_sqe = (np_interval_sqe[np_interval_sqe > 0])[1:] \
                                 - (np_interval_sqe[np_interval_sqe > 0])[:-1]
        if print_info:
            print(np_server_interval_sqe)
            print(np_client_interval_sqe)
        if np_server_interval_sqe.shape[0] == 0:
            np_server_interval_sqe = np.array([0])
        if np_client_interval_sqe.shape[0] == 0:
            np_client_interval_sqe = np.array([0])
        features.append(np_client_interval_sqe.max()) # client包到达时间最大值、最小值、平均数、方差29 30 31 32
        features.append(np_client_interval_sqe.min())
        features.append(np_client_interval_sqe.mean())
        features.append(np_client_interval_sqe.var())
        features.append(np_server_interval_sqe.max()) # server包到达时间最大值、最小值、平均数、方差33 34 35 36
        features.append(np_server_interval_sqe.min())
        features.append(np_server_interval_sqe.mean())
        features.append(np_server_interval_sqe.var())
        output_file.write(str(features)[1:-1])

        load_layer("tls")

        if ch[2:5] in [b'\x01\x03\x01',b'\x01\x03\x00',b'\x01\x00\x02']:
            ch = TLS(ch)
        else:
            ch_len = struct.pack('i',len(ch))
            ch = TLS(b'\x16\x03\x03' + ch_len[1:2] + ch_len[0:1] + ch)
        sh_len = struct.pack('i', len(sh))
        sh = TLS(b'\x16\x03\x03' + sh_len[1:2] + sh_len[0:1] + sh)

        if print_info:
            ch.show()
            sh.show()

        features = list()
        features.append(sh.msg[0].version) # tls版本37
        features.append(sh.msg[0].cipher) # 提供的加密套件列表38
        features.append(len(ch.msg[0].ciphers)) # 加密套件数39

        output_file.write(str(features)[1:-1])
        output_file.write(',\"' + str(ch.msg[0].ciphers)[1:-1] + '\",')

        # feature 39

        features = list()

        if ch.haslayer('SSLv2'):
            features.append(0)
            features.append(0)
            output_file.write(str(features)[1:-1])
            output_file.write(',\"\",')
        else:
            if ch.msg[0].ext:
                features.append(len(ch.msg[0].ext)) # 40
                features.append(ch.msg[0].extlen) # 41
            else:
                features.append(0)
                features.append(0)
            output_file.write(str(features)[1:-1])

            ext_names = ''
            if ch.msg[0].ext:
                for ext in ch.msg[0].ext:
                    ext_names += (ext.name + ',')
                output_file.write(',\"' + ext_names[:-1] + '\",')
            else:
                output_file.write(',\"\",')
            # feature 42

        features = list()
        if sh.msg[0].ext:
            features.append(len(sh.msg[0].ext)) # feature 43
            features.append(sh.msg[0].extlen) # feature 44
        else:
            features.append(0)
            features.append(0)
        output_file.write(str(features)[1:-1])

        ext_names = ''
        if sh.msg[0].ext:
            for ext in sh.msg[0].ext:
                ext_names += (ext.name + ',')
            output_file.write(',\"' + ext_names[:-1] + '\",')
        else:
            output_file.write(',\"\",')
        # feature 45

        output_file.write('\n')

