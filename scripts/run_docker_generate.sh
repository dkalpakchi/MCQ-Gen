#!/bin/bash
MODEL=$1
MODEL_DIR=$2
CHECKPOINT_ID=$3
END_ARGS=$4

prompts=(
  "Ett minne från brinnande pandemi, ett par månader in, hela världen hade fått upp ögonen för Sveriges coronahantering, presskonferenserna fylldes av internationell press som häpet stod på rad och undrade vad vi egentligen höll på med. Varför stänger ni inte landet? Varför slår ni inte igen skolorna, som alla andra? Varför håller ni idrottshallar och restauranger öppna? Ingen fattade vad vi sysslade med, och vi förlöjligades och hånades, särskilt statsepidemiolog Anders Tegnell fick klä skott, han blev den globala portalfiguren för slarvig coronahantering."
  "Maria är 20 år, och Johan är dubbelt så gammal"
  "Region Stockholm vill minska notan för bemanningspersonal, som under de senaste åren skenat till miljardbelopp. Mellan 2021 och 2022 växte utgiftsposten med drygt 150 miljoner kronor.
  Under mars påbörjar regionen att fasa ut konsultsjuksköterskor på alla sjukhus utom Karolinska universitetssjukhuset som fått dispens. Planen är att den inhyrda personalstyrkan ska minska med minst en tredjedel fram till april. Hur regionen gör därefter är inte beslutat men tanken är att minskningen ska fortsätta. Flera sjukhus uppger att en minskning av hyrpersonalen är en av de viktigaste åtgärderna om man ska klara budgeten för 2023."
  "Stockholm är Sveriges huvudstad."
  "Kyiv är Ukrainas huvudstad."
  "Jakob Byggmestere, även känd som Jacob Richter, död 1571, var en tysk-svensk arkitekt och konstsnickare verksam i Sverige. Jakob byggmestere härstammade från Freiburg och anställdes 1540 av Gustav Vasa som byggmästare vid slottet Tre Kronor i Stockholm och från början av 1550-talet även vid Kalmar slott där stora delen av den inre konstnärliga utsmyckningen utfördes av honom. Från 1561 ledde han arbetet med stadens befästningar och vid byggandet av Kronobergs slott."
)

for ((i = 0; i < ${#prompts[@]}; i++)) do
	docker run --rm -t --shm-size=10g -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v=$(pwd):/workspace --name dmytroka_swectrl --gpus all --cpus=10 -m=60g qg_exp bash -c "cd /workspace; python -m models.$MODEL.generate -f $MODEL_DIR/checkpoint-$CHECKPOINT_ID -fa $MODEL_DIR/ft_args.bin -p '${prompts[$i]}' $END_ARGS"
done

