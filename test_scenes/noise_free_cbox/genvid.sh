# ffmpeg -y -gamma 2.2 -r 20 -i ./in/beauty_%d.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-input.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_0_reproject_kernel.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-output_0.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_0_overall_shiftvec.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-shiftvec-output_0.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_0_final_alpha.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-final_alpha-output_0.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_0_dist_after_shift.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-dist_after_shift-output_0.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_0_merge_kernel_subpixel.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-merge_kernel_subpixel-output_0.mp4

ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_1_reproject_kernel.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-output_1.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_1_overall_shiftvec.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-shiftvec-output_1.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_1_final_alpha.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-final_alpha-output_1.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_1_dist_after_shift.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-dist_after_shift-output_1.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_1_merge_kernel_subpixel.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-merge_kernel_subpixel-output_1.mp4

ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_2_reproject_kernel.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-output_2.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_2_overall_shiftvec.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-shiftvec-output_2.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_2_final_alpha.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-final_alpha-output_2.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_2_dist_after_shift.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-dist_after_shift-output_2.mp4
ffmpeg -y -gamma 2.2 -r 20 -i ./out/frame%d_hlevel_2_merge_kernel_subpixel.exr -vcodec libx264 -pix_fmt yuv420p -preset slow -crf 18 -r 25 box-noisefree-merge_kernel_subpixel-output_2.mp4
