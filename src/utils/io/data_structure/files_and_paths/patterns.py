import re


# ======================== General Path Operations ========================
# Date has the following format: yyyy.mm.dd or yy_mm_dd at the beginning of the file name
DATE_PATTERN = re.compile(r"\b(\d{4}|\d{2})(\.|\_)(\d{2})(\.|\_)(\d{2})")
# Pattern for the subject id
SUBJECT_ID_PATTERN = re.compile(r"\bhen-i(\d){3}\b|\bHEN-I(\d){3}\b|\bsubj_(\d){3}\b|Cells|Cells_(\d){2}\b")
# Pattern of weeks folder should always be at least one digit followed by "_weeks"
WEEKS_PATTERN = re.compile(r"\b(\d+)\_weeks\b")
# Roi consists of a  3-digits number
ROI_PATTERN = re.compile(r"\b\d{3}\b")
# Channel name should start with the channel followed by the number of the channel
CHANNEL_NAME_STRING = r"channel_(\d+)"
CHANNEL_PATTERN = re.compile(rf"\b{CHANNEL_NAME_STRING}\b")
# Label folder name should with the treshold then the smoothing sigma and the volume thresholds
LFN_STRING = r"t_(\d+\.\d+)_s_(\d+)_v_(\d+)_(\d+|inf)\b|manual|ground_truth"
LABEL_FOLDER_NAME_PATTERN = re.compile(rf"\b{LFN_STRING}$")
# Label file name should start with the date then the roi
ROI_FROM_NIFTI_FILENAME_PATTERN = re.compile(r"\b\d{4}\_\d{2}\_\d{2}\_(\d{3})")
# Registration folder name should start with registration_to_ followed by the number of weeks
REGISTRATION_FOLDER_NAME_PATTERN = re.compile(r"^registration_to_(\d+_weeks)$")
# Matching channel a (settings) to channel b (settings) folder name follow the pattern
# channel_a_(settings)_to_channel_b_(settings)
CHANNEL_A_RELATED_TO_CHANNEL_B_FILE_NAME = re.compile(
    rf"\b{CHANNEL_NAME_STRING}_related_to_{CHANNEL_NAME_STRING}_timeseries.json$"
)
MATCHING_CHANNEL_A_SET_TO_CHANNEL_B_SET_FOLDER_NAME = re.compile(
    rf"\b{CHANNEL_NAME_STRING}_\({LFN_STRING}\)_{CHANNEL_NAME_STRING}_\({LFN_STRING}\)$"
)
GRAPH_LINEAGE_FOLDER_NAME = r"lineage_graphs"
GRAPH_LINEAGE_FILE_NAME = re.compile(
    rf"\b{SUBJECT_ID_PATTERN}_{ROI_PATTERN}_graph.graphml"
)
