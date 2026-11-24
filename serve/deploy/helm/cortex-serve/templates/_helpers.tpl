{{/* Common labels */}}
{{- define "cortex.labels" -}}
app.kubernetes.io/part-of: cortex-serve
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/* Name of a model's Deployment/Service */}}
{{- define "cortex.modelName" -}}
cortex-model-{{ . }}
{{- end }}

{{/* Merge a model entry with llamacpp.defaults: model keys win */}}
{{- define "cortex.modelArg" -}}
{{- $m := index . 0 -}}{{- $d := index . 1 -}}{{- $k := index . 2 -}}
{{- if hasKey $m $k }}{{ index $m $k }}{{ else }}{{ index $d $k }}{{ end -}}
{{- end }}

{{/* True if a model entry is enabled (default true) */}}
{{- define "cortex.modelEnabled" -}}
{{- if or (not (hasKey . "enabled")) .enabled }}true{{ end -}}
{{- end }}

{{- define "cortex.gatewayName" -}}cortex-gateway{{- end }}
{{- define "cortex.dbName" -}}cortex-gateway-db{{- end }}
