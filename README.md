# 🎮 Tactic 🤖

A collection of interactive games combining physical hardware control and computer vision through a modern web interface.

> Developed as a project for the Advanced Microprocessors course.

> *This project is dedicated to our incredible team of collaborators whose innovation, persistence, and teamwork made it all possible.*

## 🔍 Overview

Tactic is a platform that integrates computer vision, machine learning, and robotic control to create engaging physical games that can be played through a web interface. The system uses a camera to observe the physical game state, AI to understand what's happening, and controls robotic components via an ESP32 microcontroller to manipulate physical game pieces.

## 💻 Technologies Used

### Backend
- **FastAPI**: High-performance Python web framework for building APIs
- **OpenCV**: Computer vision library for image processing and analysis
- **YOLOv5**: Real-time object detection for identifying game pieces
- **WebSockets**: Real-time bidirectional communication
- **asyncio**: Asynchronous I/O for handling concurrent operations
- **NumPy**: Scientific computing library for numerical operations

### Hardware Control
- **ESP32**: Microcontroller for controlling physical game components
- **Arduino**: Programming environment for the ESP32
- **Servo Motors**: For precise mechanical movements
- **Stepper Motors**: For rotation and positioning
- **Relays**: For controlling different game components

### Frontend
- **Next.js**: React framework for building the web interface
- **React**: JavaScript library for building user interfaces
- **TailwindCSS**: Utility-first CSS framework
- **WebSockets**: Client-side implementation for real-time communication

## 🏗️ System Architecture

```
┌─────────────┐     WebSockets     ┌─────────────┐     WebSockets     ┌─────────────┐
│             │<──────────────────>│             │<──────────────────>│             │
│  Next.js    │    JSON Messages   │  FastAPI    │    JSON Commands   │    ESP32    │
│  Frontend   │                    │  Backend    │                    │  Controller │
│             │<───────────────────│             │                    │             │
└─────────────┘    Video Frames    └─────────────┘                    └─────────────┘
                                        │                                    │
                                        │                                    │
                                  ┌─────▼─────┐                        ┌─────▼─────┐
                                  │ Computer  │                        │ Physical  │
                                  │ Vision &  │                        │ Hardware  │
                                  │ AI Models │                        │ Components│
                                  └───────────┘                        └───────────┘
```

## 📡 WebSocket Communication

The project implements a three-layer WebSocket communication system:

1. **Frontend to Backend WebSockets**:
   - Each game connects to a specific WebSocket endpoint (`ws://localhost:8000/ws/{game_id}`)
   - Sends configuration data and (for some games) camera frames to the backend
   - Receives game state updates, processed frames, and arm movement status

2. **Backend to ESP32 WebSockets**:
   - The backend maintains a WebSocket connection to the ESP32 (`ws://192.168.168.84:80`)
   - Sends JSON commands to control game-specific hardware components
   - Implements auto-reconnect and command retry logic for reliable operation

3. **ESP32 Internal WebSocket Server**:
   - The ESP32 runs a WebSocket server that listens for commands
   - Handles game switching, servo control, and motor movements
   - Returns success/failure status for each command

## 🎲 Games

### Memory Matching 🧠
A vision-based memory matching game with robotic arm interaction. The system detects card positions, manipulates cards with a robotic arm, and uses either color detection or YOLO object detection to identify matches.

### Tic Tac Toe ❌⭕
A classic game where a robotic arm plays against the user, using computer vision to track the board state.

### Rubik's Cube Game 🟩🟥🟦
The system detects a Rubik's cube's state and controls a mechanism to solve it.

### Shell Game 🥚
A classic shell game with computer vision to track the positions of cups and a hidden ball.

### Target Shooter 🎯
A shooting game where the system uses vision to aim at targets.

## 🔌 ESP32 Integration

The ESP32 is programmed with Arduino to control various hardware components:

- Manages different game modes through relay control
- Controls servo motors for precise arm movements
- Drives stepper motors for rotation mechanisms
- Handles electromagnets for picking up game pieces

The ESP32 WebSocket server processes JSON messages with the following structure:
```json
{
  "action": "switch|command",
  "game": "ARM|SHOOTER|RUBIK",  // for "switch" action
  "command": "servo_angles_or_game_specific_command"  // for "command" action
}
```

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- ESP32 with appropriate circuitry

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
npm install
npm run dev
```

### ESP32 Setup
1. Install Arduino IDE
2. Install ESP32 board support
3. Upload the `esp32_webserver.ino` sketch to your ESP32

## 📂 Project Structure

- `/backend` - FastAPI backend server
  - `/games` - Game-specific backend logic
  - `/utils` - Utility functions including ESP32 WebSocket client
- `/src` - Next.js frontend
  - `/app` - Pages and components
  - `/app/games` - Game-specific frontend implementations

## 👨‍💻 Meet the Team 👩‍💻

This project was brought to life by a dedicated team of collaborators:

| Profile | Contributor |
|:-------:|:------------|
| <img src="https://github.com/Abdelrahman-Adel610.png" width="75" height="75" alt="Abdelrahman"> | **Abdelrahman Adel Hashiem** ([@Abdelrahman-Adel610](https://github.com/Abdelrahman-Adel610)) |
| <img src="https://github.com/ahmedfathy0-0.png" width="75" height="75" alt="Ahmed Fathy"> | **Ahmed Fathy** ([@ahmedfathy0-0](https://github.com/ahmedfathy0-0)) |
| <img src="https://github.com/ahmedGamalEllabban.png" width="75" height="75" alt="Ahmed Ellabban"> | **Ahmed Ellabban** ([@ahmedGamalEllabban](https://github.com/ahmedGamalEllabban)) |
| <img src="https://github.com/AliAlaa88.png" width="75" height="75" alt="Alieldin Alaa"> | **Alieldin Alaa** ([@AliAlaa88](https://github.com/AliAlaa88)) |
| <img src="https://github.com/Alyaa242.png" width="75" height="75" alt="Alyaa Ali"> | **Alyaa Ali** ([@Alyaa242](https://github.com/Alyaa242)) |
| <img src="https://github.com/AmiraKhalid04.png" width="75" height="75" alt="Amira Khalid"> | **Amira Khalid** ([@AmiraKhalid04](https://github.com/AmiraKhalid04)) |
| <img src="https://github.com/engmohamed-emad.png" width="75" height="75" alt="Mohamed Emad"> | **Mohamed Emad** ([@engmohamed-emad](https://github.com/engmohamed-emad)) |
| <img src="https://github.com/habibayman.png" width="75" height="75" alt="Habiba Ayman"> | **Habiba Ayman** ([@habibayman](https://github.com/habibayman)) |
| <img src="https://github.com/im-saif.png" width="75" height="75" alt="Saif"> | **Saif** ([@im-saif](https://github.com/im-saif)) |
| <img src="https://github.com/KarimZakzouk.png" width="75" height="75" alt="Karim Zakzouk"> | **Karim Zakzouk** ([@KarimZakzouk](https://github.com/KarimZakzouk)) |

We extend our heartfelt appreciation to each member of this incredible team for their dedication, creativity, and collaborative spirit in bringing Tactic to life. ❤️

## 📜 License

MIT
