import numpy as np
import cv2

########Not used
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
############

# Read and convert template image to grayscale
# For yellow template
#im = cv2.imread('./video/secuencia_1/prueba_491_color.png')
# For blue template
im = cv2.imread('./video/secuencia_1/prueba_1096_color.png')
im = im[:, 200:612]
#im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#im_gray=im_hsv[:,:,1]
#cv2.namedWindow('Gray Frame', cv2.WINDOW_NORMAL)
#cv2.imshow('Gray Frame', im_gray)
#cv2.waitKey(0)
# Crop template from the image
# For yellow template
#h = im[215:404, 64:158]  # Manually adjusted crop coordinates
#cv2.imwrite('./templates/yellow_template.png', h)

# For blue template
#h = im[170:350, 64:158]  # Manually adjusted crop coordinates
#cv2.imwrite('./templates/blue_template.png', h)


pixel_size = 3.1 / 80 #3.1cm, 80 pixels

h = cv2.imread('./templates/yellow_template.png', cv2.IMREAD_GRAYSCALE)
#h = cv2.imread('./templates/blue_template.png', cv2.IMREAD_GRAYSCALE)
#h = cv2.imread('./templates/red_template.png', cv2.IMREAD_GRAYSCALE)
#h = cv2.imread('./templates/green_template.png', cv2.IMREAD_GRAYSCALE)
#h = adjust_gamma(h, 3.5)
#h = cv2.cvtColor(h, cv2.COLOR_BGR2HSV)
#h = h[:,:,1]
# Pad template to match image size
h_padded = np.zeros_like(im_gray)
h_padded[:h.shape[0], :h.shape[1]] = h

cv2.namedWindow('Template', cv2.WINDOW_NORMAL)
cv2.imshow('Template', h_padded)
#cv2.waitKey(0)
# Perform Fourier Transform on the template
hF = np.fft.fft2(h_padded)
hF_shifted = np.fft.fftshift(hF)

# Open video
#cap = cv2.VideoCapture('./video/secuencia_1/output.mp4')
cap = cv2.VideoCapture('./video/secuencia_2/output.mp4')
#cap.set(cv2.CAP_PROP_FPS, 15)
block_count=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[:, 200:612]
    #frame = adjust_gamma(frame, 3.5)
    
    # Convert current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_gray = adjust_gamma(frame_gray, 3.5)
    # HSV frame
    #frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #frame_gray = frame_hsv[:,:,1]
    #cv2.imshow('gray frame', frame_gray)

    # Perform Fourier Transform on the current frame
    frame_F = np.fft.fft2(frame_gray)
    frame_F_shifted = np.fft.fftshift(frame_F)
    frame_F_conj = np.conj(frame_F_shifted)

    # Correlation in frequency domain
    im_final = frame_F_conj * hF_shifted
    im_final_shifted = np.fft.ifftshift(im_final)

    

    # Inverse Fourier Transform to get the spatial domain result
    imFsp = np.fft.ifft2(im_final_shifted)
    imFsp_real = np.real(imFsp)
    imFsp_real = np.rot90(imFsp_real, 2)

    # Normalize and display correlation result
    imFsp_real_norm = cv2.normalize(imFsp_real, None, 0, 255, cv2.NORM_MINMAX)
    imFsp_real_norm = np.uint8(imFsp_real_norm)
    cv2.namedWindow('Correlation Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Correlation Result', imFsp_real_norm)

    # Thresholding to find the highest correlation
    threshold = imFsp_real > np.max(imFsp_real) * 0.98
    threshold_uint8 = (threshold * 255).astype(np.uint8)
    cv2.namedWindow('Highest Correlation', cv2.WINDOW_NORMAL)
    cv2.imshow('Highest Correlation', threshold_uint8)

    # Find location of the highest correlation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imFsp_real)
    print('\nmax_val: ', max_val)
    # Drawing counting line in the top of the video
    count_pixel = 150
    line_color = (0,255,255)
    
    cv2.line(frame, (0, count_pixel), (frame.shape[1], count_pixel), line_color, 2)

    cv2.putText(frame, f"Count: {block_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #if 35000000>max_val>31000000: #For green template
    #if 40000000>max_val>35000000: #For red template
    #if 60000000>max_val>50000000: #For blue template
    if max_val>150000000: # For yellow template
        # Draw bounding box on the original frame
        top_left = max_loc
        bottom_right = (top_left[0] + h.shape[1], top_left[1] + h.shape[0])
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green bounding box

        # Draw centroid
        centroid = [(top_left[0] + bottom_right[0]) // 2, ((top_left[1]+ bottom_right[1])//2)]
        cv2.circle(frame, centroid, 1, (0, 0, 255), 2)

        # Draw clamping points at left and rigth of the bounding box
        clamping_left = (top_left[0], (top_left[1] + h.shape[0]//2))
        clamping_right = (bottom_right[0], (top_left[1] + h.shape[0]//2))
        distance_clamping_points = (clamping_right[0] - clamping_left[0]) * pixel_size
        cv2.putText(frame, f"{distance_clamping_points} cm ", (bottom_right[0] - 80, bottom_right[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame, clamping_left, 1, (0, 0, 255), 2)
        cv2.circle(frame, clamping_right, 1, (0, 0, 255), 2)

        if count_pixel - 4 < round(centroid[1])<count_pixel+4:
            line_color = (255,255,255)
            block_count += 1
            cv2.line(frame, (0, count_pixel), (frame.shape[1], count_pixel), line_color, 2)
            #centroid[1]=0.0
        #   print('Centroid: ', centroid)

   

    # Showing real frame
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.waitKey(100)
    
# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()


