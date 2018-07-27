

source activate nlp
python main.py --task_num 1 --model mlp --lr 1e-4  --epoch 20 --exp_name mlp_without_mask_slot_word_data_form_1 --keep_prob 0.8
python main.py --task_num 1 --model mlp --lr 1e-4  --epoch 20 --exp_name mlp_without_mask_slot_word_data_form_1 --keep_prob 0.8
python main.py --task_num 1 --model h_lstm --lr 1e-4  --epoch 20 --exp_name test_lr_1e-4_h_lstm_mask_input_0_real_attn_0 --keep_prob 0.8 --mask_input 0
python main.py --task_num 1 --model attn_net_data_form_1 --lr 1e-4  --epoch 20 --exp_name final_attn_net_data_form_1 --keep_prob 0.75 --mask_input 0
python main.py --task_num 1 --model lstm --lr 1e-4  --epoch 20 --exp_name bilstm_mask_input_0_attn_0 --keep_prob 0.8


python main.py --task_num 1 --model mix_model --lr 1e-4  --epoch 20 --exp_name mix_model_test --keep_prob 0.9 --mask_input 0
python main.py --task_num 1 --model attn_net_data_form_1 --lr 1e-4  --epoch 20 --exp_name test --keep_prob 0.8 --mask_input 0


python main.py --task_num 1 --model lstm --lr 1e-4  --epoch 20 --exp_name test --keep_prob 0.8


