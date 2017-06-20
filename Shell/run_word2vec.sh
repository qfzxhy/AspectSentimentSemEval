DIMS = "30 50 100 150 200 250 300"
for length in $DIMS
do
    time make run-word2vec datafile=./yelp_reveiw.txt size = ${length}
done