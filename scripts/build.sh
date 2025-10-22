rm -rf build && 
rm -rf dist && 
rm -rf linghe.egg-info &&
python setup.py develop && 
python setup.py bdist_wheel 

