export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/cw/liir_code/NoCsBack/vgaraujo/TextClassificationDatasets
export OUTPUT_DIR=/cw/liir_code/NoCsBack/vgaraujo/ESP-CL/NLP
export ORDER=1

echo "Running order $ORDER on GPU $CUDA_VISIBLE_DEVICES"
if [ "$ORDER" == "1" ]; then
TASKS="${DATA_DIR}/yelp_review_full_csv ${DATA_DIR}/ag_news_csv ${DATA_DIR}/dbpedia_csv ${DATA_DIR}/amazon_review_full_csv ${DATA_DIR}/yahoo_answers_csv"
elif [ "$ORDER" == "2" ]; then
TASKS="${DATA_DIR}/dbpedia_csv ${DATA_DIR}/yahoo_answers_csv ${DATA_DIR}/ag_news_csv ${DATA_DIR}/amazon_review_full_csv ${DATA_DIR}/yelp_review_full_csv"
elif [ "$ORDER" == "3" ]; then
TASKS="${DATA_DIR}/yelp_review_full_csv ${DATA_DIR}/yahoo_answers_csv ${DATA_DIR}/amazon_review_full_csv ${DATA_DIR}/dbpedia_csv ${DATA_DIR}/ag_news_csv"
elif [ "$ORDER" == "4" ]; then
TASKS="${DATA_DIR}/ag_news_csv ${DATA_DIR}/yelp_review_full_csv ${DATA_DIR}/amazon_review_full_csv ${DATA_DIR}/yahoo_answers_csv ${DATA_DIR}/dbpedia_csv"
fi

cd NLP

NAME="output_esp20only_${ORDER}"
python3 train.py --tasks $TASKS --output_dir $NAME --overwrite --mem_capacity 0.2 --batch_size 32 --only_mem
python3 test.py --output_dir $NAME
