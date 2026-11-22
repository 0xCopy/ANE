#define MODEL_NAME "small"
#define DIM 64
#define HIDDEN 96
#define HEADS 4
#define KV_HEADS 2
#define HD 16
#define GQA_RATIO (HEADS / KV_HEADS)
#define Q_DIM (HEADS * HD)
#define KV_DIM (KV_HEADS * HD)
#define SEQ 32
#define NLAYERS 2
#define VOCAB 10
#define CKPT_PATH "x"
#define DEFAULT_DATA_PATH "x"
