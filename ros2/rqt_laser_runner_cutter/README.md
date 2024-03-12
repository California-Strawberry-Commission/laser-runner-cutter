This rqt plugin is used for visualization. Once this package is built via colcon, run `rqt`, and under Plugins you should see Laser Runner Cutter. If the plugin does not appear in rqt, you may need to run `rqt --force-discover`.

To develop the plugin UI, install QT Designer:

        $ pip install pyqt5-tools
        $ pyqt5-tools designer

Then, open the UI file `rqt_laser_runner_cutter.ui`.
