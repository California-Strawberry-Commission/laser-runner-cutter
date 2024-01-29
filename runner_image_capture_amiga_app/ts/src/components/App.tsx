import CameraPreview from "@/components/CameraPreview";
import { Button } from "@/components/ui/button";

function App() {
  return (
    <div className="p-4">
      <header className="pb-4">
        <h1 className="text-3xl font-bold">Runner Image Capture</h1>
      </header>
      <main className="flex flex-col gap-4">
        <div className="flex flex-row gap-4">
          <Button>Save Frame</Button>
          <Button>Save Frame</Button>
          <Button>Save Frame</Button>
        </div>
        <CameraPreview />
      </main>
    </div>
  );
}

export default App;
