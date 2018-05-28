# DSN 									*** flag for dsn training and dsn transfer, NOT FOR dsn eval
DSN = True

# WebVision1000
# WebVision500A_Google
# WebVision500B_Google
# WebVision500A_Flickr
# WebVision500B_Flickr
# WebVision_Transfer_AG-BF				*** only for SOURCE & EVAL
# WebVision_Transfer_AF-BG				*** only for SOURCE & EVAL
# DSN
# DSN_fix2								*** only for SOURCE & EVAL
# DSN_fix5								*** only for SOURCE & EVALs

SOURCE = "WebVision500A_Google" 
TARGET = "WebVision500B_Flickr"
EVAL   = ""

#conv_1/
#conv_3/
#conv_5/
#conv_6/
#conv_7/
#fc8/
#fc9/

FIX_LAYER = ["conv_1/", "conv_3/", "conv_5/", "conv_6/", "conv_7/", "fc8/", "fc9/"]
