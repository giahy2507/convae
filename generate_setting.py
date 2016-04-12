__author__ = 'HyNguyen'
import os



def generate_settint_duc_2004():
    model_path = "data/model"
    peer_path = "data/peer"
    rouge_model_path = "model"
    rouge_peer_path = "peer"
    file_model_names = os.listdir(model_path)
    file_peer_names = os.listdir(peer_path)

    counter  = 0
    config_file = '<ROUGE_EVAL version="1.55">'
    for file_name in file_peer_names:
        file_id,_,_,_,_ = file_name.split(".")
        files_model_name_by_id = [file_model_name for file_model_name in file_model_names if file_model_name.find(file_id) != -1]
        config_file += '\t<EVAL ID="' + str(counter) + '">\n'
        config_file += '\t\t<MODEL-ROOT>\n'
        config_file += '\t\t\t'+rouge_model_path+'\n'
        config_file += '\t\t</MODEL-ROOT>\n'
        config_file += '\t\t<PEER-ROOT>\n'
        config_file += '\t\t\t'+rouge_peer_path+'\n'
        config_file += '\t\t</PEER-ROOT>\n'
        config_file += '\t\t<INPUT-FORMAT TYPE="SPL">\n'
        config_file += '\t\t</INPUT-FORMAT>\n'
        config_file += '\t\t<PEERS>\n'
        config_file += '\t\t\t<P ID="1">'+file_name +'</P>\n'
        config_file += '\t\t</PEERS>\n'
        config_file += '\t\t<MODELS>\n'
        for file_model_name_by_id in files_model_name_by_id:
            _,_,_,_,model_id = file_model_name_by_id.split(".")
            config_file += '\t\t\t<M ID="'+model_id+'">'+file_model_name_by_id+'</M>\n'
        config_file += '\t\t</MODELS>\n'
        config_file += '\t</EVAL>\n\n'
        counter +=1
    config_file += '</ROUGE_EVAL>\n'
    file_settings = open('data/settings.xml','w')
    file_settings.write(config_file)
    file_settings.close()

if __name__ == "__main__":


    model_path = "data/model"
    peer_path = "data/peer"
    rouge_model_path = "model"
    rouge_peer_path = "peer"
    folder_model_names = os.listdir(model_path)
    folder_peer_names = os.listdir(peer_path)

    counter  = 0
    config_file = '<ROUGE_EVAL version="1.55">'
    for folder_name in folder_peer_names:
        config_file += '\t<EVAL ID="' + str(counter) + '">\n'
        config_file += '\t\t<MODEL-ROOT>\n'
        config_file += '\t\t\t'+rouge_model_path+"/"+folder_name +'\n'
        config_file += '\t\t</MODEL-ROOT>\n'
        config_file += '\t\t<PEER-ROOT>\n'
        config_file += '\t\t\t'+rouge_peer_path+"/"+folder_name +'\n'
        config_file += '\t\t</PEER-ROOT>\n'
        config_file += '\t\t<INPUT-FORMAT TYPE="SPL">\n'
        config_file += '\t\t</INPUT-FORMAT>\n'
        config_file += '\t\t<PEERS>\n'
        config_file += '\t\t\t<P ID="1">'+folder_name+".1.txt"+'</P>\n'
        config_file += '\t\t</PEERS>\n'
        config_file += '\t\t<MODELS>\n'
        for file_model in os.listdir(model_path+"/"+folder_name):
            _,model_id,_ = file_model.split(".")
            config_file += '\t\t\t<M ID="'+model_id+'">'+file_model+'</M>\n'
        config_file += '\t\t</MODELS>\n'
        config_file += '\t</EVAL>\n\n'
        counter +=1
    config_file += '</ROUGE_EVAL>\n'
    file_settings = open('data/setting.xml','w')
    file_settings.write(config_file)
    file_settings.close()