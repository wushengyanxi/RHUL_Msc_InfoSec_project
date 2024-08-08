model = ['originp','responp', 'flow_duration', 'fwd_pkts_tot',
         'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
         'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
         'down_up_ratio', 'fwd_header_size_tot', 'fwd_header_size_min', 
         'fwd_header_size_max', 'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max',
         'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count', 
         'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count',
         'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count', 
         'fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg',
         'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot',
         'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min', 'flow_pkts_payload.max',
         'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min',
         'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max',
         'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot',
         'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts',
         'bwd_subflow_pkts', 'fwd_subflow_bytes', 'bwd_subflow_bytes', 'fwd_bulk_bytes',
         'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate',
         'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std',
         'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std', 'fwd_init_window_size',
         'bwd_init_window_size', 'fwd_last_window_size', 'Label']

heaviest_features = ['originp','responp', 'flow_duration', 'fwd_pkts_tot',
         'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
         'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
         'down_up_ratio', 'fwd_header_size_tot', 'fwd_header_size_min', 
         'fwd_header_size_max', 'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max',
         'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count', 
         'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count',
         'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count', 
         'fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg',
         'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot',
         'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min', 'flow_pkts_payload.max',
         'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min',
         'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max',
         'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot',
         'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts',
         'bwd_subflow_pkts', 'bwd_subflow_bytes', 'fwd_bulk_bytes',
         'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate',
         'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std',
         'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std',
         'bwd_init_window_size', 'fwd_last_window_size', 'Label']

print(heaviest_features[30])
print(heaviest_features[28])

G1_10000 = [['flow_CWR_flag_count', 1.8866090808028044], ['fwd_URG_flag_count', 1.7251598054463462], ['bwd_pkts_payload.avg', 1.7148554407081604], ['fwd_pkts_per_sec', 1.6688908018735522], ['flow_pkts_payload.max', 1.6268036878263175], ['flow_pkts_per_sec', 1.6184658977945339], ['bwd_pkts_per_sec', 1.5488778453260226], ['fwd_pkts_payload.min', 1.5363351876869185], ['responh', 1.458247255472429], ['fwd_header_size_min', 1.4125100559001849], ['bwd_pkts_tot', 1.3824238555869257], ['bwd_bulk_packets', 1.3582998004626308], ['originp', 1.3560683654000165], ['fwd_bulk_bytes', 1.3369930014829765], ['bwd_URG_flag_count', 1.3314196089514518], ['flow_iat.tot', 1.3104510932923836], ['fwd_iat.std', 1.2792812584771425], ['bwd_iat.tot', 1.2481260366144018], ['flow_FIN_flag_count', 1.239928964547359], ['bwd_bulk_rate', 1.2362001147646535], ['flow_ACK_flag_count', 1.1745866389682322], ['originh', 1.1715269391668628], ['down_up_ratio', 1.1588498219058478], ['bwd_data_pkts_tot', 1.135868500731811], ['fwd_pkts_payload.tot', 1.1158674356204588]]
G2_10000 = [['bwd_pkts_payload.avg', 2.121642840944189], ['flow_CWR_flag_count', 1.9445688393407532], ['fwd_pkts_per_sec', 1.6954053903207602], ['fwd_URG_flag_count', 1.6633917943706336], ['originh', 1.6595311315430625], ['bwd_pkts_per_sec', 1.5940389471260703], ['fwd_pkts_payload.min', 1.4863767255499567], ['fwd_iat.std', 1.4749524086769579], ['bwd_pkts_payload.min', 1.4571026964289457], ['fwd_header_size_min', 1.4526323677377675], ['active.max', 1.411894563243431], ['bwd_URG_flag_count', 1.3302969445876358], ['fwd_header_size_max', 1.299642876073954], ['flow_pkts_payload.tot', 1.2956245421608377], ['uid', 1.2875656251573875], ['bwd_iat.tot', 1.2607453491564073], ['flow_duration', 1.2532061965255692], ['active.min', 1.2076760081662847], ['down_up_ratio', 1.1887970687631493], ['bwd_pkts_payload.tot', 1.1760718257487806], ['bwd_PSH_flag_count', 1.1460659027804607], ['responh', 1.1456888561048717], ['fwd_pkts_tot', 1.099824325846668], ['flow_RST_flag_count', 1.0986097300530562], ['originp', 1.0747053634138863]]
G3_15000 = [['bwd_pkts_payload.avg', 1.8824274911886192], ['flow_CWR_flag_count', 1.8452547348765056], ['fwd_URG_flag_count', 1.513380355036314], ['fwd_pkts_per_sec', 1.42850801068112], ['bwd_pkts_per_sec', 1.4184696980444058], ['bwd_iat.avg', 1.3604892442449616], ['bwd_pkts_payload.min', 1.248343321545511], ['bwd_pkts_payload.tot', 1.23431812988694], ['bwd_iat.min', 1.194761071053533], ['fwd_header_size_min', 1.1863947607099146], ['fwd_pkts_payload.min', 1.1734717763166074], ['flow_pkts_payload.avg', 1.160755102967586], ['bwd_PSH_flag_count', 1.1521915240399596], ['active.min', 1.146535304997521], ['fwd_iat.std', 1.1405899256784853], ['down_up_ratio', 1.0946842037448556], ['bwd_header_size_min', 1.0477081476656218], ['fwd_header_size_max', 1.011736814555576], ['bwd_URG_flag_count', 1.0019226438521691], ['flow_FIN_flag_count', 0.9651684849352105], ['originh', 0.9594533908987675], ['fwd_pkts_payload.avg', 0.9248789870094001], ['uid', 0.8627022292942244], ['fwd_bulk_bytes', 0.8554168873738555], ['flow_iat.max', 0.8515462986763511]]







