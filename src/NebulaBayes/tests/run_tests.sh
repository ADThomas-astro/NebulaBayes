# Shell script to run NebulaBayes testing

# python2 ../dereddening.py
# python3 ../dereddening.py

echo
echo
echo "========================"
echo "Testing with python 2..."
python2 test_NB.py
echo
echo
echo
echo "========================"
echo "Testing with python 3..."
python3 test_NB.py

echo "Tests finished"
