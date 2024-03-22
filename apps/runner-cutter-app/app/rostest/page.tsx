import Dashboard from "@/components/rostest/dashboard";

export default function RosTest() {
  return (
    <main className="flex flex-col min-h-screen items-center justify-center p-4 gap-4">
      <h1 className="text-5xl font-bold text-center">ROS Test</h1>
      <Dashboard />
    </main>
  );
}
