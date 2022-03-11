mkdir -p logs

echo "Predicting..."
python main.py "$@" | tee logs/main.out

echo "Submission created!"