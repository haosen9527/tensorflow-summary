#include "tensorflow/contrib/tensorboard/db/summary_file_writer.h"

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "iostream"

using namespace std;

using namespace tensorflow;

using namespace tensorflow::ops;

class FakeClockEnv : public EnvWrapper {
 public:
  FakeClockEnv() : EnvWrapper(Env::Default()), current_millis_(0) {}
  void AdvanceByMillis(const uint64 millis) { current_millis_ += millis; }
  uint64 NowMicros() override { return current_millis_ * 1000; }
  uint64 NowSeconds() override { return current_millis_ * 1000; }

 private:
  uint64 current_millis_;
};


bool (*func_ptr)(string &str);
bool str_size(string &str)
{
    return str.size();
}

bool func_pointer ()
{
   // func_ptr = str_size;
    func_ptr = &str_size;
    string name = "name";
    return (*func_ptr)(name);
}



int main()
{
  FakeClockEnv env_;
  SummaryWriterInterface* writer;
  CreateSummaryFileWriter(1, 1, "./", "test_name", &env_, &writer);
  Scope scope = Scope::NewRootScope();
  MatMul matmul(scope, Const(scope, {{1, 2}}), Const(scope, {{3}, {4}}));
  GraphDef *graph = new GraphDef();
  scope.ToGraphDef(graph);
  ClientSession session(scope);
  std::vector<Tensor> outputs;
  session.Run({matmul}, &outputs);
  std::cout << outputs[0].DebugString() << std::endl;
  writer->WriteGraph(2, std::unique_ptr<GraphDef>(graph));
  writer->Flush();

    cout<<"bool:"<<func_pointer()<<endl;

  return 0;
}
