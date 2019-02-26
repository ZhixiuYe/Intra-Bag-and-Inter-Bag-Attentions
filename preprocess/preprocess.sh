g++ extract.cpp -o extract -O2 -lpthread
./extract
python data2pkl.py
python pickledata.py
rm *txt
rm extract
rm *temp*
cd ..
if [ ! -d "result" ]; then
  mkdir result
fi
