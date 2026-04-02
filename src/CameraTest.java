import java.awt.Robot;
import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.InputEvent;

import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class CameraTest {
    public static void main(String[] args) {

        // Load OpenCV DLL
        System.load("C:\\Users\\Nithin NRJ\\Downloads\\opencv\\build\\java\\x64\\opencv_java4120.dll");

        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("Camera not detected!");
            return;
        }

        Mat frame = new Mat();
        Mat prevFrame = new Mat();

        // Robot for mouse control
        Robot robot;
        try {
            robot = new Robot();
        } catch (AWTException e) {
            System.out.println("Robot error");
            return;
        }

        // Screen size
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        int screenWidth = (int) screenSize.getWidth();
        int screenHeight = (int) screenSize.getHeight();

        // Smoothing variables
        int prevMouseX = 0;
        int prevMouseY = 0;

        // Click control variables
        long lastClickTime = 0;
        int prevCenterX = 0;
        int prevCenterY = 0;

        while (true) {
            camera.read(frame);

            // Convert to grayscale
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            // Blur image
            Imgproc.GaussianBlur(gray, gray, new Size(21, 21), 0);

            if (!prevFrame.empty()) {

                // Compute difference
                Mat diff = new Mat();
                Core.absdiff(prevFrame, gray, diff);

                // Threshold + improve detection
                Imgproc.threshold(diff, diff, 15, 255, Imgproc.THRESH_BINARY);
                Imgproc.dilate(diff, diff, new Mat(), new Point(-1, -1), 2);

                // Find contours
                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(diff, contours, new Mat(),
                        Imgproc.RETR_EXTERNAL,
                        Imgproc.CHAIN_APPROX_SIMPLE);

                for (MatOfPoint contour : contours) {
                    if (Imgproc.contourArea(contour) > 2000) {

                        Rect rect = Imgproc.boundingRect(contour);

                        int centerX = rect.x + rect.width / 2;
                        int centerY = rect.y + rect.height / 2;

                        // Draw center point
                        Imgproc.circle(frame, new Point(centerX, centerY), 5, new Scalar(0, 0, 255), -1);

                        // Draw rectangle
                        Imgproc.rectangle(frame, rect, new Scalar(0, 255, 0), 2);

                        // 🔹 Move mouse
                        int mouseX = (int) ((double) centerX / frame.width() * screenWidth);
                        int mouseY = (int) ((double) centerY / frame.height() * screenHeight);

                        // Smooth movement
                        int smoothX = (prevMouseX + mouseX) / 2;
                        int smoothY = (prevMouseY + mouseY) / 2;

                        robot.mouseMove(smoothX, smoothY);

                        prevMouseX = smoothX;
                        prevMouseY = smoothY;

                        // 🔹 CLICK LOGIC (stable hand)
                        int dx = Math.abs(centerX - prevCenterX);
                        int dy = Math.abs(centerY - prevCenterY);

                        if (dx < 10 && dy < 10) {
                            long currentTime = System.currentTimeMillis();

                            if (currentTime - lastClickTime > 1000) {
                                robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
                                robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);

                                lastClickTime = currentTime;
                            }
                        }

                        // Update previous center
                        prevCenterX = centerX;
                        prevCenterY = centerY;
                    }
                }
            }

            // Show output
            HighGui.imshow("Motion Tracking", frame);

            // Save frame
            gray.copyTo(prevFrame);

            // Exit on ESC
            if (HighGui.waitKey(30) == 27) {
                break;
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
}
