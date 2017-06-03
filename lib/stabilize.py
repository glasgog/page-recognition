import numpy as np

MIN_MATCH = 10
NEEDED_FRAME = 15
MAX_STRENGTH = int(NEEDED_FRAME / 2)  # deve essere MAX_STRENGTH < NEEDED_FRAME

DEBUG = True


def suspicious(ref_img, features, index, sospetto, strength):
    # TODO: posso ritornare -1 invece di strenght quando si verifica qualche tipo di errore
    # TODO: nota che nel programma ref_img non e' solo il nome ma e'
    # l'immagine in se'. non e' necessario passarla come argomento
    if DEBUG:
        # print " input: " + str(ref_img[index]) +  " with matches " +
        # str(features)
        print " input: image index " + str(index) + " with matches " + str(features)
    if features >= MIN_MATCH:
        for i in range(len(sospetto)):
            if i == index:
                if sospetto[i] < NEEDED_FRAME:
                    sospetto[i] += 1
                # in realta' non e' mai maggiore. Non uso else perche' deve
                # essere eseguito subito dopo l'incremento
                if sospetto[i] >= NEEDED_FRAME:
                    if strength == 0:
                        strength = MAX_STRENGTH
                    elif not strength == MAX_STRENGTH:  # altrimenti viene incrementata all'infinito quando c'e' una sequenza di frame con lo stesso ref
                        strength += 1
            elif sospetto[i] > 0:
                if sospetto[i] < NEEDED_FRAME:
                    sospetto[i] -= 1
                else:
                    if strength > 0:
                        strength -= 1
                        print "\033[93m" + "decreasing strength" + "\033[0m"
                    else:
                        sospetto[i] = NEEDED_FRAME - MAX_STRENGTH
    else:
        if strength > 0:
            strength -= 1
            print "\033[93m" + "decreasing strength" + "\033[0m"
        else:
            pass  # CHECK: cosa dovrebbe fare? ridurre tutti i sospetti?.
            for i in range(len(sospetto)):
                if sospetto[i] > 0:
                    sospetto[i] -= 1
            #sospetto[i] = NEEDED_FRAME - MAX_STRENGTH
    if max(sospetto) >= NEEDED_FRAME:
        i_best = np.argmax(sospetto)  # indice del miglior sospetto
        if DEBUG:
            # print " output: " + str(ref_img[i_best]) + " with strenght " +
            # str(strength)
            print " output: image index " + str(i_best) + " with strenght " + str(strength)
    else:
        # se non e' stato trovato un buon sospetto restituisco la lunghezza dell'array in modo che,
        # nel caso non venga effettuato un check sul valore e si richiami
        # ref_img[i_best], venga generato un errore
        i_best = len(ref_img)
        if DEBUG:
            print " no output and strenght " + str(strength)

    # NOTA: strenght e' immutabile ed e' quindi necessario restituirla, mentre sospetto non lo e'
    # ricordare che i_best e' maggiore della lunghezza dell'array a cui fa
    # riferimento
    return i_best, strength


def main():
    ref_img = ["page1_ref.jpg", "page2_ref.jpg", "page3_ref.jpg"]  # TMP
    sospetto = len(ref_img) * [0]

    # TMP | ometto lunghezza best_good per ora
    best_match_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # best_good_match_list = 15*[[30,0]] #
    # [[30,0],[30,0],[30,0],[30,0],[30,0],...] | [num features, reference
    # match]
    best_good_match_list = 15 * [[30, 0]] + 10 * [[30, 1]] + 15 * [[30, 0]]
    strength = 0
    debug_count = 0
    for features, index in best_good_match_list:  # TMP | indica il best match trovato per ogni frame
        print "\niteration " + str(debug_count)
        debug_count += 1
        best_candidate_index, strength = suspicious(
            ref_img, features, index, sospetto, strength)
        # print strength

    # print sospetto


if __name__ == '__main__':
    main()
