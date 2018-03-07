find ./sourcecode/ -type f -not -path "*/__pycache__/*" | while read line ; do
    chmod +x $line
    PYTHONPATH=./sourcecode python $line
done