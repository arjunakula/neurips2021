import pickle
with open('/media/4TB/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/neurips2021/parrot.pkl', 'rb') as f:
    samples = pickle.load(f)

single_scores = []

cnt_diff_1 = 0
cnt_diff_2 = 0
cnt_diff_3 = 0
cnt_diff_4 = 0
cnt_diff_5 = 0
cnt_diff_6 = 0
cnt_diff_7 = 0
cnt_diff_8 = 0
cnt_diff_9 = 0
cnt_diff_10 = 0

cnt = 0

for i in range(0,len(samples['real_string_lengths'])):
    #print(i)
    # pgm_score = (samples['real_pgm_score'][i] - min(samples['real_pgm_score']))*1.0/(max(samples['real_pgm_score'])- min(samples['real_pgm_score']))
    # spatLen = (samples['real_string_spatialRel_cnt'][i]- min(samples['real_string_spatialRel_cnt']))*1.0/(max(samples['real_string_spatialRel_cnt'])-min(samples['real_string_spatialRel_cnt']))
    # pgm_len = (samples['real_pgm_lengths'][i] - min(samples['real_pgm_lengths']))*1.0/(max(samples['real_pgm_lengths']) - min(samples['real_pgm_lengths']))
    # strLen = (samples['real_string_lengths'][i]-min(samples['real_string_lengths']))*1.0/(max(samples['real_string_lengths']) - min(samples['real_string_lengths']))

    pgm_score = (samples['real_pgm_score'][i] - 1)*1.0/(999- 1)
    spatLen = (samples['real_string_spatialRel_cnt'][i]- 0)*1.0/(6-0)
    pgm_len = (samples['real_pgm_lengths'][i] - 4)*1.0/(24 - 4)
    strLen = (samples['real_string_lengths'][i]-4)*1.0/(61 - 4)


    total_score= (7.0*pgm_score + 1.0*spatLen + 1.0*pgm_len + 1.0*strLen) * 1.0/10.0 
 
    if(total_score <= 0.04):
        #if('union' in samples['real_programs'][i] or 'unique' in )
        #if('filter_color[cyan]' in samples['real_programs'][i] and 'filter_size[large]' in samples['real_programs'][i]):
        #print(samples['real_programs'][i]+"\n")
        #print("pgm_Score:"+str(pgm_score))
        #print("strLen_Score:"+str(strLen))
        cnt_diff_1 = cnt_diff_1 + 1
    elif(total_score > 0.04 and total_score <= 0.06):
        #print(samples['real_programs'][i]+"\n")
        # if('filter_color[cyan]' in samples['real_programs'][i] and 'filter_size[large]' in samples['real_programs'][i]):
        #     print(samples['real_programs'][i]+"\n")
        cnt_diff_2 = cnt_diff_2 + 1
    elif(total_score > 0.06 and total_score <= 0.07):
        #print(samples['real_programs'][i]+"\n")
        # if( 'filter_ordinal' in samples['real_programs'][i] and 'filter_color[cyan]' in samples['real_programs'][i] and 'filter_size[large]' in samples['real_programs'][i]):
        #     print(samples['real_programs'][i]+"\n")
        cnt_diff_3 = cnt_diff_3 + 1
    elif(total_score > 0.07 and total_score <= 0.1):
        #print(samples['real_programs'][i]+"\n")
        #print(samples['real_programs'][i]+"\n")
        cnt_diff_4 = cnt_diff_4 + 1
    elif(total_score > 0.1 and total_score <= 0.2):
        #print(samples['real_programs'][i]+"\n")
        cnt_diff_5 = cnt_diff_5 + 1
    elif(total_score > 0.2 and total_score <= 0.26):
        #print(samples['real_programs'][i]+"\n")
        cnt_diff_6 = cnt_diff_6 + 1
    elif(total_score > 0.26 and total_score <= 0.28):
        #print(samples['real_programs'][i]+"\n")
        cnt_diff_7 = cnt_diff_7 + 1
    elif(total_score > 0.28 and total_score <= 0.4):
        #print(samples['real_programs'][i]+"\n")
        # cnt = cnt + 1
        # if(cnt <= 5):
        #     print("<Exp>: "+samples['real_strings'][i])
        #     print("<Pgm>: "+samples['real_programs'][i])
        #     print("\n")
        cnt_diff_8 = cnt_diff_8 + 1
    elif(total_score > 0.4 and total_score <= 0.5):
        if('metallic object(s)' in samples['real_strings'][i] and 'front' in samples['real_strings'][i] and  'filter_ordinal' in samples['real_programs'][i] and 'filter_color[cyan]' in samples['real_programs'][i]):
            print("<Exp>: "+samples['real_strings'][i])
            print("<Pgm>: "+samples['real_programs'][i])
        #print(samples['real_programs'][i]+"\n")
        # cnt = cnt + 1
        # if(cnt <= 5):
        #     print("<Exp>: "+samples['real_strings'][i])
        #     print("<Pgm>: "+samples['real_programs'][i])
        #     print("\n")
        cnt_diff_9 = cnt_diff_9 + 1
    elif(total_score > 0.5):
        
        #cnt = cnt + 1
        # if(cnt <= 5):
        #     print("<Exp>: "+samples['real_strings'][i])
        #     print("<Pgm>: "+samples['real_programs'][i])
        #     print("\n")
        #print(samples['real_programs'][i]+"\n")
        # cnt = cnt + 1
        # if(cnt <= 5):
        #     print("<Exp>: "+samples['real_strings'][i])
        #     print("<Pgm>: "+samples['real_programs'][i])
        #     print("\n")
        cnt_diff_10 = cnt_diff_10 + 1

print(cnt_diff_1)
print(cnt_diff_2)
print(cnt_diff_3)
print(cnt_diff_4)
print(cnt_diff_5)
print(cnt_diff_6)
print(cnt_diff_7)
print(cnt_diff_8)
print(cnt_diff_9)
print(cnt_diff_10)


