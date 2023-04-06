import function
from scapy.all import *
load_layer("tls")

if __name__ == "__main__":

    data_path = '../test/'
    class_names = ['test1', 'test2']
    # data_path = 'Dataset/'
    # class_names = ['Dridex', 'Normal', 'Bunitu', 'Miuref', 'TrickBot', 'HTBot']


    for j,class_name in enumerate(class_names):

        print(j + 1, class_name)
        session_names = os.listdir(data_path + class_name + '/sessions/')
        output_path = data_path + 'features/' + class_name + '_features.csv'
        sessions_len = len(session_names)

        with open(output_path, 'w') as file_object:
            pass

        record = [0, 0]
        for i,session_name in enumerate(session_names):

            print(i+1,session_name)
            if i % 20  == 9:
                print('\nfinished ' + str(round(i/sessions_len*100,3)) + '%')
                print(class_name + ' : ' +str(record)[1:-1] + '\n')
            pcapfile = data_path + class_name + '/sessions/' + session_name
            e = function.get_features(pcapfile,output_path)

            if e == 0:
                record[0] += 1
            else:
                record[1] += 1

