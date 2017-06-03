import numpy as np

MIN_MATCH = 10
NEEDED_FRAME = 10
MAX_STRENGTH = 5 # deve essere MAX_STRENGTH < NEEDED_FRAME

def main():
	ref_img = ["page1_ref.jpg", "page2_ref.jpg", "page3_ref.jpg"] #TMP
	sospetto = len(ref_img)*[0]

	best_match_index = [0,0,0,0,0,0,0,0,0,0,0,0,0] #TMP | ometto lunghezza best_good per ora
	#best_good_match_list = 15*[[30,0]] # [[30,0],[30,0],[30,0],[30,0],[30,0],...] | [num features, reference match]
	best_good_match_list = 15*[[30,0]]+10*[[30,1]]+15*[[30,0]]
	strength = 0
	for features, index in best_good_match_list: #TMP | indica il best match trovato per ogni frame
		print("input:", ref_img[index], "with matches", features)
		if features > MIN_MATCH:
			for i in range(len(sospetto)):
				if i == index:
					if sospetto[i] < NEEDED_FRAME:
						sospetto[i] += 1
					if sospetto[i] >= NEEDED_FRAME:	#in realta' non e' mai maggiore. Non uso else perche' deve essere eseguito subito dopo l'incremento
						if strength == 0:
							strength = MAX_STRENGTH
						elif not strength == MAX_STRENGTH: #altrimenti viene incrementata all'infinito quando c'e' una sequenza di frame con lo stesso ref
							strength += 1
				elif sospetto[i] > 0:
					if sospetto[i] < NEEDED_FRAME:
						sospetto[i] -= 1
					else:
						if strength > 0:
							strength -= 1
						else:
							sospetto[i] = NEEDED_FRAME - MAX_STRENGTH
		else:
			if strength > 0:
				strength -= 1
			else:
				sospetto[i] = NEEDED_FRAME - MAX_STRENGTH
		if max(sospetto) >= NEEDED_FRAME:
			i = np.argmax(sospetto)
			print(" ", ref_img[i], strength)
		else:
			print(" ", "nope", strength)



if __name__ == '__main__':
    main()
