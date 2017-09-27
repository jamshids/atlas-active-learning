MYDIR="/common/external/rawabd/Mindboggle-101/NRRD/"
RESULTDIR="/common/external/rawabd/Mindboggle-101/CC/Results"

# listing all the image files in an array
fnames=(/common/external/rawabd/Mindboggle-101/NRRD/*)
printf "%s\n" "${fnames[@]}" > results/image_list.txt


for i in ${!fnames[@]}
do 
    # writing the file if necessary
    if [ ! -e "results/$i.txt" ] ; then
	touch "results/$i.txt"
	
	# put 100 empty line in the file
	for t in {0..99}; do echo -en '\n' >> results/$i.txt; done
    fi

    for ((j=$i;j<=${#fnames[@]};j++))
    do
	# do not re-compute cross-correlations if it is already computed
	if [ -e "results/details/cc_$i-$j.txt" ] ; then
	    cc=$( tail -n 1 results/details/cc_$i-$j.txt )
	    # writing it in its location
	    L=$((j+1))
	    sed -i "${L}s/.*/${cc}/" results/$i.txt
	    continue
	fi
	
	echo ""
	echo "*******************************************************************"
	echo "Computing cross-correlations between image $i and $j"
	echo "image 1: ${fnames[$i]}"
	echo "image 2: ${fnames[$j]}"
	echo "*******************************************************************"
	echo ""

	./lcc ${fnames[$i]} ${fnames[$j]} 2 2 2 1 10 5
	
	mv lcc.txt results/details/cc_$i-$j.txt
	mv index.txt results/details/index_$i-$j.txt
	
	# store the CC
	cc=$( tail -n 1 results/details/cc_$i-$j.txt )
	L=$((j+1))
	sed -i "${L}s/.*/${cc}/" results/$i.txt
    done
done