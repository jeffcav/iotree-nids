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
| 1 | `MIN_TTL` | 0.115425 |
| 2 | `TCP_WIN_MAX_IN` | 0.079855 |
| 3 | `SRC_TO_DST_SECOND_BYTES` | 0.055096 |
| 4 | `L4_DST_PORT` | 0.053812 |
| 5 | `L7_PROTO` | 0.052959 |
| 6 | `L4_SRC_PORT` | 0.046133 |
| 7 | `DST_TO_SRC_AVG_THROUGHPUT` | 0.045053 |
| 8 | `DST_TO_SRC_SECOND_BYTES` | 0.043114 |

### Mode: integer

| Rank | Feature | Mean Importance |
|-----:|:--------|----------------:|
| 1 | `MIN_TTL` | 0.128717 |
| 2 | `L4_DST_PORT` | 0.068992 |
| 3 | `TCP_WIN_MAX_IN` | 0.067393 |
| 4 | `MIN_IP_PKT_LEN` | 0.058129 |
| 5 | `LONGEST_FLOW_PKT` | 0.049014 |
| 6 | `L4_SRC_PORT` | 0.046097 |
| 7 | `MAX_TTL` | 0.041235 |
| 8 | `DNS_QUERY_TYPE` | 0.039628 |

## Per-Dataset Top Features

Each column shows the top features ranked by mean PD-based importance within that dataset independently.

### Mode: float

| Rank | NF-BoT-IoT-v2 | NF-CSE-CIC-IDS2018-v2 | NF-ToN-IoT-v2 | NF-UNSW-NB15-v2 |
|-----:|:--------------|:----------------------|:--------------|:----------------|
| 1 | `MIN_TTL` (0.1886) | `MAX_TTL` (0.1179) | `SRC_TO_DST_SECOND_BYTES` (0.1020) | `MIN_TTL` (0.2440) |
| 2 | `MIN_IP_PKT_LEN` (0.1417) | `L4_DST_PORT` (0.1102) | `DNS_TTL_ANSWER` (0.0912) | `TCP_WIN_MAX_IN` (0.1587) |
| 3 | `IN_BYTES` (0.1350) | `RETRANSMITTED_IN_BYTES` (0.0573) | `L7_PROTO` (0.0907) | `LONGEST_FLOW_PKT` (0.0610) |
| 4 | `DST_TO_SRC_SECOND_BYTES` (0.1172) | `L4_SRC_PORT` (0.0555) | `FLOW_DURATION_MILLISECONDS` (0.0885) | `ICMP_TYPE` (0.0243) |
| 5 | `DST_TO_SRC_AVG_THROUGHPUT` (0.1155) | `LONGEST_FLOW_PKT` (0.0440) | `TCP_WIN_MAX_IN` (0.0754) | `SRC_TO_DST_SECOND_BYTES` (0.0209) |
| 6 | `SRC_TO_DST_SECOND_BYTES` (0.0975) | `DURATION_IN` (0.0433) | `DNS_QUERY_TYPE` (0.0707) | `FLOW_DURATION_MILLISECONDS` (0.0201) |
| 7 | `SHORTEST_FLOW_PKT` (0.0895) | `L7_PROTO` (0.0374) | `L4_SRC_PORT` (0.0632) | `DST_TO_SRC_SECOND_BYTES` (0.0163) |
| 8 | `DNS_QUERY_ID` (0.0875) | `TCP_WIN_MAX_IN` (0.0360) | `NUM_PKTS_UP_TO_128_BYTES` (0.0616) | `SERVER_TCP_FLAGS` (0.0155) |

### Mode: integer

| Rank | NF-BoT-IoT-v2 | NF-CSE-CIC-IDS2018-v2 | NF-ToN-IoT-v2 | NF-UNSW-NB15-v2 |
|-----:|:--------------|:----------------------|:--------------|:----------------|
| 1 | `MIN_IP_PKT_LEN` (0.2262) | `L4_DST_PORT` (0.1357) | `DNS_QUERY_TYPE` (0.1092) | `MIN_TTL` (0.2438) |
| 2 | `MIN_TTL` (0.2222) | `MAX_TTL` (0.1100) | `DNS_TTL_ANSWER` (0.0998) | `TCP_WIN_MAX_IN` (0.1584) |
| 3 | `DST_TO_SRC_SECOND_BYTES` (0.1275) | `RETRANSMITTED_IN_BYTES` (0.0557) | `FLOW_DURATION_MILLISECONDS` (0.0796) | `LONGEST_FLOW_PKT` (0.0592) |
| 4 | `DNS_QUERY_ID` (0.1092) | `L4_SRC_PORT` (0.0514) | `TCP_WIN_MAX_IN` (0.0771) | `SRC_TO_DST_SECOND_BYTES` (0.0239) |
| 5 | `NUM_PKTS_256_TO_512_BYTES` (0.1031) | `DURATION_IN` (0.0440) | `SRC_TO_DST_SECOND_BYTES` (0.0752) | `ICMP_TYPE` (0.0218) |
| 6 | `IN_BYTES` (0.1022) | `LONGEST_FLOW_PKT` (0.0345) | `LONGEST_FLOW_PKT` (0.0682) | `FLOW_DURATION_MILLISECONDS` (0.0209) |
| 7 | `RETRANSMITTED_OUT_BYTES` (0.0913) | `MIN_TTL` (0.0313) | `L4_SRC_PORT` (0.0610) | `L4_DST_PORT` (0.0136) |
| 8 | `OUT_BYTES` (0.0910) | `TCP_FLAGS` (0.0185) | `TCP_WIN_MAX_OUT` (0.0565) | `SERVER_TCP_FLAGS` (0.0134) |
