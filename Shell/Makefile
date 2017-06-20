run-word2vec:
    cd ../word2vec; ./word2vec -train ${datafile} -output vectors-${size}.txt -cbow 0 -size ${size} -window 5 -negative 5 -hs 0 -sample 1e-3 -binary 0
    @echo "[INFO] word2vec is finished"