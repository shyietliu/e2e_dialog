

source activate nlp
python main.py --task_num 1 --model mlp --lr 1e-4  --epoch 20 --exp_name mlp_without_mask_slot_word_data_form_1 --keep_prob 0.8 --save_model False
python main.py --task_num 1 --model h_lstm --lr 1e-4  --epoch 20 --exp_name h_lstm_without_masking --keep_prob 0.8 --save_model False
python main.py --task_num 1 --model attn_net_data_form_1 --lr 1e-4  --epoch 20 --exp_name attn_net_data_form_1 --keep_prob 0.8 --save_model False
python main.py --task_num 1 --model lstm --lr 1e-4  --epoch 20 --exp_name lstm_mask_slot_word --keep_prob 0.8 --save_model False



gcloud compute scp instance-2:/home/shyietliu/e2e_dialog/exp_log/task1/h_lstm_without_masking/log/h_lstm_without_masking_log.txt ../exp_log/h_lstm_without_masking_log.txt