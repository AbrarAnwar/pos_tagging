python buildtagger.py sents.train test.model
python runtagger.py sents.test test.model ~/Desktop/CS4248/pa1/output.txt
python mistakes.py output.txt sents.answer test.model > test.txt
