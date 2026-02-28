# PD-based Feature Importance

Importance measure: **standard deviation of the partial dependence function** (Greenwell, Boehmke & McCarthy, 2018, arXiv:1805.04755).

## Parameters

| Parameter | Value |
|:----------|:------|
| Modes | float, integer |
| Models aggregated | 168 |
| Top-N | 8 |

## Overall Ranking

### Mode: float

| Rank | Feature | Mean Importance |
|-----:|:--------|----------------:|
| 1 | `MIN_TTL` | 0.115948 |
| 2 | `TCP_WIN_MAX_IN` | 0.079363 |
| 3 | `SRC_TO_DST_SECOND_BYTES` | 0.055298 |
| 4 | `L4_DST_PORT` | 0.054195 |
| 5 | `L7_PROTO` | 0.052847 |
| 6 | `L4_SRC_PORT` | 0.046232 |
| 7 | `DST_TO_SRC_AVG_THROUGHPUT` | 0.044455 |
| 8 | `DST_TO_SRC_SECOND_BYTES` | 0.042998 |

### Mode: integer

| Rank | Feature | Mean Importance |
|-----:|:--------|----------------:|
| 1 | `MIN_TTL` | 0.118041 |
| 2 | `L4_DST_PORT` | 0.071945 |
| 3 | `TCP_WIN_MAX_IN` | 0.064893 |
| 4 | `MIN_IP_PKT_LEN` | 0.059989 |
| 5 | `MAX_TTL` | 0.052850 |
| 6 | `IN_BYTES` | 0.051581 |
| 7 | `LONGEST_FLOW_PKT` | 0.050869 |
| 8 | `L4_SRC_PORT` | 0.047109 |

## Per-Dataset Top Features

Each column shows the top features ranked by mean PD-based importance within that dataset independently.

### Mode: float

| Rank | NF-BoT-IoT-v2 | NF-CSE-CIC-IDS2018-v2 | NF-ToN-IoT-v2 | NF-UNSW-NB15-v2 |
|-----:|:--------------|:----------------------|:--------------|:----------------|
| 1 | `MIN_TTL` (0.1913) | `MAX_TTL` (0.1179) | `SRC_TO_DST_SECOND_BYTES` (0.1023) | `MIN_TTL` (0.2437) |
| 2 | `MIN_IP_PKT_LEN` (0.1411) | `L4_DST_PORT` (0.1109) | `DNS_TTL_ANSWER` (0.0914) | `TCP_WIN_MAX_IN` (0.1587) |
| 3 | `IN_BYTES` (0.1346) | `RETRANSMITTED_IN_BYTES` (0.0572) | `L7_PROTO` (0.0900) | `LONGEST_FLOW_PKT` (0.0646) |
| 4 | `DST_TO_SRC_SECOND_BYTES` (0.1169) | `L4_SRC_PORT` (0.0553) | `FLOW_DURATION_MILLISECONDS` (0.0877) | `ICMP_TYPE` (0.0243) |
| 5 | `DST_TO_SRC_AVG_THROUGHPUT` (0.1130) | `LONGEST_FLOW_PKT` (0.0437) | `TCP_WIN_MAX_IN` (0.0754) | `SRC_TO_DST_SECOND_BYTES` (0.0210) |
| 6 | `SRC_TO_DST_SECOND_BYTES` (0.0979) | `DURATION_IN` (0.0432) | `DNS_QUERY_TYPE` (0.0710) | `FLOW_DURATION_MILLISECONDS` (0.0201) |
| 7 | `SHORTEST_FLOW_PKT` (0.0902) | `L7_PROTO` (0.0371) | `L4_SRC_PORT` (0.0641) | `DST_TO_SRC_SECOND_BYTES` (0.0163) |
| 8 | `DNS_QUERY_ID` (0.0872) | `TCP_WIN_MAX_IN` (0.0358) | `NUM_PKTS_UP_TO_128_BYTES` (0.0617) | `SERVER_TCP_FLAGS` (0.0155) |

### Mode: integer

| Rank | NF-BoT-IoT-v2 | NF-CSE-CIC-IDS2018-v2 | NF-ToN-IoT-v2 | NF-UNSW-NB15-v2 |
|-----:|:--------------|:----------------------|:--------------|:----------------|
| 1 | `MIN_IP_PKT_LEN` (0.2342) | `L4_DST_PORT` (0.1348) | `DNS_QUERY_TYPE` (0.1098) | `MIN_TTL` (0.2242) |
| 2 | `MIN_TTL` (0.2159) | `MAX_TTL` (0.1096) | `DNS_TTL_ANSWER` (0.0983) | `TCP_WIN_MAX_IN` (0.1588) |
| 3 | `IN_BYTES` (0.1832) | `RETRANSMITTED_IN_BYTES` (0.0557) | `NUM_PKTS_UP_TO_128_BYTES` (0.0823) | `LONGEST_FLOW_PKT` (0.0636) |
| 4 | `DNS_QUERY_ID` (0.1074) | `L4_SRC_PORT` (0.0516) | `MAX_TTL` (0.0821) | `FLOW_DURATION_MILLISECONDS` (0.0199) |
| 5 | `L4_DST_PORT` (0.1034) | `DURATION_IN` (0.0442) | `FLOW_DURATION_MILLISECONDS` (0.0777) | `MAX_TTL` (0.0197) |
| 6 | `DURATION_OUT` (0.0999) | `LONGEST_FLOW_PKT` (0.0345) | `LONGEST_FLOW_PKT` (0.0715) | `SERVER_TCP_FLAGS` (0.0187) |
| 7 | `NUM_PKTS_256_TO_512_BYTES` (0.0733) | `MIN_TTL` (0.0299) | `TCP_WIN_MAX_IN` (0.0684) | `OUT_BYTES` (0.0159) |
| 8 | `L4_SRC_PORT` (0.0714) | `TCP_WIN_MAX_IN` (0.0185) | `L4_SRC_PORT` (0.0601) | `L4_DST_PORT` (0.0121) |
