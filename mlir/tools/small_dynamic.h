#define MODEL_NAME "small"
#define DIM 8
#define HIDDEN 12
#define HEADS 4
#define KV_HEADS 2
#define HD 4
#define GQA_RATIO (HEADS / KV_HEADS)
#define Q_DIM (HEADS * HD)
#define KV_DIM (KV_HEADS * HD)
#define SEQ 6
#define NLAYERS 2
#define VOCAB 10
#define CKPT_PATH "x"
#define DEFAULT_DATA_PATH "x"
