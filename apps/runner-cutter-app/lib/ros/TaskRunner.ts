export default class TaskRunner {
  private taskFunction: () => any;
  private intervalMs: number;
  private cancelExecution: boolean;
  private isRunning: boolean;

  constructor(taskFunction: () => any, intervalMs: number) {
    this.taskFunction = taskFunction;
    this.intervalMs = intervalMs;
    this.cancelExecution = false;
    this.isRunning = false;
  }

  public async start(): Promise<void> {
    if (this.isRunning) {
      return;
    }

    this.cancelExecution = false;
    this.isRunning = true;

    while (!this.cancelExecution) {
      try {
        await this.taskFunction();
      } catch (error) {
        console.error("Error during task execution:", error);
      }
      await new Promise((resolve) => setTimeout(resolve, this.intervalMs));
    }

    this.isRunning = false;
  }

  public stop(): void {
    if (!this.isRunning) {
      return;
    }

    this.cancelExecution = true;
  }
}
