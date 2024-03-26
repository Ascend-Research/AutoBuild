# Defines constants for this search space

# PN and MBV3 constants
OFA_PN_STAGE_N_BLOCKS = (4, 4, 4, 4, 4, 1)

OFA_MBV3_STAGE_N_BLOCKS = (4, 4, 4, 4, 4)

OFA_W_PN = 1.3

OFA_W_MBV3 = 1.2

PN_BLOCKS = (
    "mbconv2_e3_k3",
    "mbconv2_e3_k5",
    "mbconv2_e3_k7",
    "mbconv2_e4_k3",
    "mbconv2_e4_k5",
    "mbconv2_e4_k7",
    "mbconv2_e6_k3",
    "mbconv2_e6_k5",
    "mbconv2_e6_k7",
)

MBV3_BLOCKS = (
    "mbconv3_e3_k3",
    "mbconv3_e3_k5",
    "mbconv3_e3_k7",
    "mbconv3_e4_k3",
    "mbconv3_e4_k5",
    "mbconv3_e4_k7",
    "mbconv3_e6_k3",
    "mbconv3_e6_k5",
    "mbconv3_e6_k7",
)

PN_OP2IDX = {}
for __i, __op in enumerate(PN_BLOCKS):
    PN_OP2IDX[__op] = __i
PN_IDX2OP = {v: k for k, v in PN_OP2IDX.items()}
assert len(PN_OP2IDX) == len(PN_IDX2OP)

MBV3_OP2IDX = {}
for __i, __op in enumerate(MBV3_BLOCKS):
    MBV3_OP2IDX[__op] = __i
MBV3_IDX2OP = {v: k for k, v in MBV3_OP2IDX.items()}
assert len(MBV3_OP2IDX) == len(MBV3_IDX2OP)

PN_NET_CONFIGS_EXAMPLE = [
    ["mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7",],
    ["mbconv2_e6_k7", "mbconv2_e6_k7", ],
    ["mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7",],
    ["mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7",],
    ["mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7",],
    ["mbconv2_e6_k7"],
]

MBV3_NET_CONFIGS_EXAMPLE = [
    ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    ["mbconv3_e6_k7", "mbconv3_e6_k7",],
    ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
]

# ResNet constants
OFA_RES_STAGE_MAX_N_BLOCKS = (4, 4, 6, 4)
OFA_RES_STAGE_MIN_N_BLOCKS = (2, 2, 4, 2)
OFA_RES_STAGE_BASE_CHANNELS = (256, 512, 1024, 2048)
OFA_RES_N_SEARCHABLE_STAGES = len(OFA_RES_STAGE_MAX_N_BLOCKS) + 1 # Stem considered a special stage
OFA_RES_WIDTH_MULTIPLIERS = (0.65, 0.8, 1.0)
OFA_RES_W_INDS = (0, 1, 2)
OFA_RES_EXPANSION_RATIOS = (0.2, 0.25, 0.35)
OFA_RES_ADDED_DEPTH_LIST = (0, 1, 2)
OFA_RES_STEM_DEPTH_LIST = (0, 2)
OFA_RES_KERNEL_SIZES = (3,)
OFA_RES_STEM_PREFIXES = ("stem+res", "stem")

# Target networks for cons acc search
PN_CONS_ACC_SEARCH_TARGET_NETS = []
assert len(set(str(__v) for __v in PN_CONS_ACC_SEARCH_TARGET_NETS)) \
       == len(PN_CONS_ACC_SEARCH_TARGET_NETS), "Target nets cannot have duplicates"

MBV3_CONS_ACC_SEARCH_TARGET_NETS = []
assert len(set(str(__v) for __v in MBV3_CONS_ACC_SEARCH_TARGET_NETS)) \
       == len(MBV3_CONS_ACC_SEARCH_TARGET_NETS), "Target nets cannot have duplicates"

RES_CONS_ACC_SEARCH_TARGET_NETS = []
assert len(set(str(__v) for __v in RES_CONS_ACC_SEARCH_TARGET_NETS)) \
       == len(RES_CONS_ACC_SEARCH_TARGET_NETS), "Target nets cannot have duplicates"

OFA_RES_STAGE_MIN_N_BLOCKS_MAX_ACC = (2, 2, 6, 4)

# Custom max blocks per stage
OFA_PN_STAGE_N_BLOCKS_GPU = (3, 3, 3, 4, 4, 1)

OFA_PN_STAGE_N_BLOCKS_CPU = (3, 3, 4, 4, 4, 1)

OFA_MBV3_STAGE_N_BLOCKS_N10 = (3, 4, 4, 4, 4)

OFA_MBV3_STAGE_N_BLOCKS_GPU = (4, 3, 4, 3, 3)

OFA_MBV3_STAGE_N_BLOCKS_CPU = (3, 3, 3, 4, 4)


# Manually designed per-stage search spaces
PN_STAGE_WISE_BLOCK_CANDS_NPU = [
    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
    ), # Stage 1 - Dropped all k7

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
    ),  # Stage 2 - Dropped all k7

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
    ), # Stage 3 - Dropped all k7

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
    ),  # Stage 4 - Dropped all k7

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
    ),  # Stage 5 - Dropped all k7

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
    ),  # Stage 6 - Dropped all k7
]

PN_STAGE_WISE_BLOCK_CANDS_GPU = [
    (
        "mbconv2_e3_k5",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ), # Stage 1 - Remove e3k3, e3k7, e4k3

    (
        "mbconv2_e3_k5",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 2 - Remove e3k3, e3k7, e4k3

    (
        "mbconv2_e3_k5",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ), # Stage 3 - Remove e3k3, e3k7, e4k3

    (
        "mbconv2_e3_k5",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 4 - Remove e3k3, e3k7, e4k3

    (
        "mbconv2_e3_k5",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 5 - Remove e3k3, e3k7, e4k3

    (
        "mbconv2_e3_k5",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 6 - Remove e3k3, e3k7, e4k3
]

PN_STAGE_WISE_BLOCK_CANDS_CPU = [
    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e3_k7",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ), # Stage 1

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e3_k7",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 2

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e3_k7",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ), # Stage 3

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e3_k7",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 4

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e3_k7",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 5

    (
        "mbconv2_e3_k3",
        "mbconv2_e3_k5",
        "mbconv2_e3_k7",
        "mbconv2_e4_k3",
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 6
]


PN_STAGE_WISE_BLOCK_CANDS_MAX_ACC = [
    (
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ), # Stage 1 - Dropped all e3 and e4k3

    (
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 2 - Dropped all e3 and e4k3

    (
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ), # Stage 3 - Dropped all e3 and e4k3

    (
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 4 - Dropped all e3 and e4k3

    (
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 5 - Dropped all e3 and e4k3

    (
        "mbconv2_e4_k5",
        "mbconv2_e4_k7",
        "mbconv2_e6_k3",
        "mbconv2_e6_k5",
        "mbconv2_e6_k7",
    ),  # Stage 6 - Dropped all e3 and e4k3
]

# MBV3
MBV3_STAGE_WISE_BLOCK_CANDS_NPU = [
    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
    ), # Stage 1 - Dropped all k7

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
    ),  # Stage 2 - Dropped all k7

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
    ), # Stage 3 - Dropped all k7

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
    ),  # Stage 4 - Dropped all k7

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
    ),  # Stage 5 - Dropped all k7
]

MBV3_STAGE_WISE_BLOCK_CANDS_N10 = [
    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 1 - Dropped e3k7, e4k7

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 2 - Dropped e3k7, e4k7

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 3 - Dropped e3k7, e4k7

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 4 - Dropped e3k7, e4k7

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 5 - Dropped e3k7, e4k7
]

MBV3_STAGE_WISE_BLOCK_CANDS_GPU = [
    (
        "mbconv3_e3_k5",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 1 - Dropped e3k3, e3k7, e4k3

    (
        "mbconv3_e3_k5",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 2 - Dropped e3k3, e3k7, e4k3

    (
        "mbconv3_e3_k5",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 3 - Dropped e3k3, e3k7, e4k3

    (
        "mbconv3_e3_k5",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 4 - Dropped e3k3, e3k7, e4k3

    (
        "mbconv3_e3_k5",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 5 - Dropped e3k3, e3k7, e4k3
]

MBV3_STAGE_WISE_BLOCK_CANDS_CPU = [
    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e3_k7",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 1

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e3_k7",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 2

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e3_k7",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 3

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e3_k7",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 4

    (
        "mbconv3_e3_k3",
        "mbconv3_e3_k5",
        "mbconv3_e3_k7",
        "mbconv3_e4_k3",
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ),  # Stage 5
]

MBV3_STAGE_WISE_BLOCK_CANDS_MAX_ACC = [
    (
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 1 - Dropped all e3 and e4k3

    (
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 2 - Dropped all e3 and e4k3

    (
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 3 - Dropped all e3 and e4k3

    (
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 4 - Dropped all e3 and e4k3

    (
        "mbconv3_e4_k5",
        "mbconv3_e4_k7",
        "mbconv3_e6_k3",
        "mbconv3_e6_k5",
        "mbconv3_e6_k7",
    ), # Stage 5 - Dropped all e3 and e4k3
]


# Manually designed per-stage mutation probs
PN_STAGE_WISE_MUTATE_PROBS_NPU = [0.25, 0.25, 0.25, 0.5, 0.5, 0.5]

PN_STAGE_WISE_MUTATE_PROBS_GPU = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

PN_STAGE_WISE_MUTATE_PROBS_CPU = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

PN_STAGE_WISE_MUTATE_PROBS_MAX_ACC = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

MBV3_STAGE_WISE_MUTATE_PROBS_NPU = [0.25, 0.25, 0.25, 0.5, 0.5]

MBV3_STAGE_WISE_MUTATE_PROBS_N10 = [0.5, 0.5, 0.5, 0.5, 0.5]

MBV3_STAGE_WISE_MUTATE_PROBS_GPU = [0.5, 0.5, 0.5, 0.5, 0.5]

MBV3_STAGE_WISE_MUTATE_PROBS_CPU = [0.5, 0.5, 0.5, 0.5, 0.5]

MBV3_STAGE_WISE_MUTATE_PROBS_MAX_ACC = [0.5, 0.5, 0.5, 0.5, 0.5]


# For OFA-ResNet
OFA_RES_STAGE_WISE_WIDTH_INDS_MAX_ACC = [
    (0, 1, 2), # For the stem
    (0, 1, 2),
    (0, 1, 2),
    (2,),
    (2,),
]
