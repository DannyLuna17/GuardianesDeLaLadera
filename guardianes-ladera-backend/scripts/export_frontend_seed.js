const fs = require("fs");
const path = require("path");
const vm = require("vm");

const projectRoot = path.resolve(__dirname, "..", "..");
const frontendRoot = path.join(projectRoot, "guardianes-ladera-frontend");
const backendRoot = path.join(projectRoot, "guardianes-ladera-backend");
const typescript = require(path.join(frontendRoot, "node_modules", "typescript"));

const mockDataPath = path.join(frontendRoot, "src", "mockData.ts");
const outputPath = path.join(backendRoot, "app", "data", "frontend_seed.json");

const source = fs.readFileSync(mockDataPath, "utf8");
const transpiled = typescript.transpileModule(source, {
  compilerOptions: {
    module: typescript.ModuleKind.CommonJS,
    target: typescript.ScriptTarget.ES2020,
  },
}).outputText;

const moduleRef = { exports: {} };
const sandbox = {
  module: moduleRef,
  exports: moduleRef.exports,
  require,
  console,
};

vm.runInNewContext(transpiled, sandbox, { filename: "mockData.js" });
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(moduleRef.exports, null, 2));
console.log(`Seed exported to ${outputPath}`);
