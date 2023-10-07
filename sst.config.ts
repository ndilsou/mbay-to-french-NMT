import { SSTConfig } from "sst";

export default {
  config(_input) {
    return {
      name: "mbay-translator",
      region: "us-east-1",
    };
  },
  stacks(app) {},
} satisfies SSTConfig;
