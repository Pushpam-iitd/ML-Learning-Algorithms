if [ $1 == '1' ]
then
	python linear2.py $2 $3 $4 $5
elif [ $1 == '2' ]
then
	python weightedLR.py $2 $3 $4
elif [ $1 == '3' ]
then
	python logistic_regression.py $2 $3
elif [ $1 == '4' ]
then
	if [ $4 == '0' ]
	then
		python gda.py $2 $3
	elif [ $4 == '1' ]
	then
		python gda2.py $2 $3
	fi
fi