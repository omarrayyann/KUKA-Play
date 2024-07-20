# Stretch-BoT

(currently) an open-vocab that assumes [concept-graph](https://concept-graphs.github.io) output as a prior

### Local CPU + Server GPU Setup

#### Installation

1. **Local CPU Setup:**

   ```sh
   # Clone the repositry
   git clone https://github.com/omarrayyann/untidy-stretch
   # Navigate to the project directory
   cd untidy-bot
   # Create a virtual environment
   python -m venv venv
   # Activate the virtual environment
   source venv/bin/activate
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Server GPU Setup:**
   ```sh
   # Clone the repositry
   git clone https://github.com/omarrayyann/untidy-stretch
   # Navigate to the server directory
   cd untidy-bot/Server/anygrasp_sdk/grasp_detection
   # Install AnyGrasp dependencies as show here: https://github.com/graspnet/anygrasp_sdk/tree/main
   # Install LangSAM dependencies as show here: https://github.com/luca-medeiros/lang-segment-anything
   ```

#### Usage

1. **Create a tunnel between the `localhost` port and the GPU server port**

   ```sh
   # Example (Using NYU Greene HPC)
   ssh -L 9875:localhost:9875 NET-ID@greene.hpc.nyu.edu
   ssh -L 9875:localhost:9875 GPU-NODE.hpc.nyu.edu

   ```

2. **Run the GPU Server**
   ```sh
   # On the GPU Node
   cd Tasks-Group/Grasping/Server/anygrasp_sdk/grasp_detection
   python server.py # make sure to adjust the port number accordingly
   ```
3. **Example Test Run the Local Setup:**

   ```sh
   python bot.py --instruction "Put the red cup on the plate" --debug --rerun
   ```

# To-Do

- [x] Mujoco Environment
- [x] Simple Navigation
- [x] ConceptGraph Scene Recorder
- [x] Inverse Kinematics
- [x] Obstacles Avoidance
- [x] Inverse Kinematics (with base)
- [x] AnyGrasp Intergation
- [x] LangSam Filtering
- [x] Rerun.io Visualization
- [x] Trajectory Obstacles-Free Grasping Filtering
- [ ] Opening drawers/closets addition (https://github.com/arjung128/stretch-open)
- [ ] Do logical instructions without instructions
