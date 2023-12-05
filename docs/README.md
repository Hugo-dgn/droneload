# DroneLoad
## Aim of the project

This project aims to create a drone with the following abilities:

- Going through a window
- Going through a tunnel
- Scanning QR codes
- Picking up objects and delivering them

During the competition, a lot of reward points are given for all the tasks that are performed auntonomously. Last year, none of the teams did any king of automation, all the drones were driven by hand.

## Developped algorithms

Currently, we have developped several algorithms, that are still to be implemented on the on bord computer (Jetson Nano):

- Autonomous window detection, optimal path computation and crossing (this can easily be adapted to the tunnel crossing)
- QR code reading
- Circles detection (in order to recognise in a picture the objects that are to be picked and delivered)

## Package installation
```bash
pip install -r requirements
```

## Project usage

cd to the destination of the main.py file

### QRCode Reading
```bash
python main.py qrcode
```
Press 'q' to exit


### Cirle detection
```bash
python main.py circles
```
Press 'q' to exit


### Contours detection
```bash
python main.py contours
```
Press 'q' to exit


### Rectangle detection
- In image issued from your camera :
```bash
python main.py vrect
```
Press 'q' to exit

- In preloaded image :
```bash
python main.py imrect path_to_image
```
Press 'q' to exit


### Find path to window
- From predefined parameters :
```bash
python main.py path
```
Press 'q' to exit

- From preloaded image
```bash
python main.py impath path_to_image
```
Press 'q' to exit

- Animate calculated path (ATTENTION PROBLEME A REGLER)
```bash
python main.py vpath
```
Press 'q' to exit
