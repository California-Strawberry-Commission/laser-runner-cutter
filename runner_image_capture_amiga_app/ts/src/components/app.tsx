import CameraPreview from "@/components/camera-preview";
import ButtonBar from "@/components/button-bar";

function App() {
  return (
    <div className="p-4">
      <main className="flex flex-row gap-4">
        <div>
          <h1 className="text-3xl font-bold pb-4">Runner Image Capture</h1>
          <ButtonBar />
        </div>
        <CameraPreview />
      </main>
    </div>
  );
}

export default App;
