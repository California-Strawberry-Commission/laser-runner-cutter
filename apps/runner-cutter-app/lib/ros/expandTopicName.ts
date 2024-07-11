import path from "path";

export default function expandTopicOrServiceName(name: string, nodeName: string, namespace: string = "/", isService: boolean = false)
{
    switch(name[0]) {
        case "~":
            return path.join(namespace, nodeName, name.slice(1))
        case "/":
            return path.normalize(name.slice(1))
        default:
            return path.join(namespace, name)
    }
}
