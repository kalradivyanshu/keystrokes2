echo "preprocessing..."
python3 preprocess.py
echo "signalizing..."
python3 signalize.py
echo "Converting to equal dims..."
python3 makeIntoEqualDims.py
echo "ConvertingToSamplesBatch..."
python3 TfconvertToSamplesBatch.py linux
