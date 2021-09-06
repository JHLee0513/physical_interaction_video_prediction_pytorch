python -u ./infer.py --data_dir /media/joonho1804/Storage1/dirt/filtered_arc_seq_dynamics_data --model CDNA --channels 1 --output_dir ./experiment_all_nodes_multi_test --sequence_length 10 --context_frames 1 --num_masks 10 --schedsamp_k -1 --batch_size 1 --learning_rate 0.001 --epochs 10 --print_interval 10 --device cuda --use_state --height 128 --width 128 --pretrained_model "./experiment_all_nodes_multi/net_epoch_9.pth"
# python -u ./infer.py --data_dir /media/joonho1804/Storage1/dirt/filtered_arc_seq_dynamics_data --model CDNA --channels 1 --output_dir ./experiment_all_nodes_test --sequence_length 10 --context_frames 1 --num_masks 10 --schedsamp_k -1 --batch_size 1 --learning_rate 0.001 --epochs 10 --print_interval 10 --device cuda --use_state --height 128 --width 128 --pretrained_model "./experiment_all_nodes/net_epoch_9.pth"
# python -u ./infer.py --data_dir ../heatmap_dynamics_data --model CDNA --channels 1 --output_dir ./weights --sequence_length 5 --context_frames 1 --num_masks 10 --schedsamp_k 900.0 --batch_size 1 --learning_rate 0.001 --epochs 10 --print_interval 10 --device cuda --use_state --height 256 --width 256 --pretrained_model "./weights/net_epoch_9.pth"
