g++ plugin_yolov3_detection_op_test.cc \
  -I ${NEUWARE_HOME}/include  \
  -I ../common/include  \
  -L ${NEUWARE_HOME}/lib64 \
  -L ../build \
  -o ./yolov3_detection_test -lcnml -lcnrt -lcnplugin --std=c++11
