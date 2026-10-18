# 주간_튜닝_작업등록.ps1 — "Weekly Tuning" 작업(매주 일요일 05:30, 주간_튜닝.bat)을 작업 스케줄러에 등록한다.
# 조건은 기존 "Daily Pipeline" 작업 설정을 읽기 전용으로 읽어 그대로 맞춘다(절전 해제 실행, 놓친 실행 보충, AC 전원, 중복 실행 무시).
# 기본은 등록하지 않고 만들 작업 내용만 보여 준다. 실제 등록은 사람이 확인한 뒤 -Register로 한다(Daily Pipeline처럼 저장된 암호 방식이면 자격 증명을 묻는다).
# 사용: powershell -ExecutionPolicy Bypass -File 주간_튜닝_작업등록.ps1 [-Register]
# 저장 형식: UTF-8(BOM) — Windows PowerShell 5.1이 한글 경로(주간_튜닝.bat)를 읽으려면 BOM이 필요하다.

param([switch]$Register)
$ErrorActionPreference = 'Stop'

$WeeklyTaskName = 'Weekly Tuning'
$ReferenceTaskName = 'Daily Pipeline'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatPath = Join-Path $ProjectRoot '주간_튜닝.bat'


function Read-ReferenceTaskSettings {
    # 설명: 기존 Daily Pipeline 작업의 조건·계정 방식을 읽어 객체로 돌려준다(읽기 전용, 작업을 바꾸지 않는다).
    # 주의 사항: 작업이 없으면 예외. 계정 방식(LogonType)이 Password면 등록 때 암호가 필요하다.
    # 입력 타입->반환 타입: () -> PSCustomObject
    $task = Get-ScheduledTask -TaskName $ReferenceTaskName
    [pscustomobject]@{
        WakeToRun                  = $task.Settings.WakeToRun
        StartWhenAvailable         = $task.Settings.StartWhenAvailable
        DisallowStartIfOnBatteries = $task.Settings.DisallowStartIfOnBatteries
        StopIfGoingOnBatteries     = $task.Settings.StopIfGoingOnBatteries
        MultipleInstances          = [string]$task.Settings.MultipleInstances
        LogonType                  = [string]$task.Principal.LogonType
        UserId                     = $task.Principal.UserId
    }
}


function New-WeeklyTuningTaskParts {
    # 설명: 참고 설정으로 Weekly Tuning 작업의 동작·트리거·설정 객체를 만든다(등록은 하지 않음).
    # 주의 사항: 트리거는 매주 일요일 05:30. 실행 제한 2시간(튜닝 약 20분 + 일일 실행 대기 최대 30분).
    # 입력 타입->반환 타입: (PSCustomObject) -> hashtable
    param($Reference)
    $settingArgs = @{ ExecutionTimeLimit = (New-TimeSpan -Hours 2); MultipleInstances = 'IgnoreNew' }
    if ($Reference.WakeToRun) { $settingArgs.WakeToRun = $true }
    if ($Reference.StartWhenAvailable) { $settingArgs.StartWhenAvailable = $true }
    if (-not $Reference.DisallowStartIfOnBatteries) { $settingArgs.AllowStartIfOnBatteries = $true }
    if (-not $Reference.StopIfGoingOnBatteries) { $settingArgs.DontStopIfGoingOnBatteries = $true }
    @{
        Action   = New-ScheduledTaskAction -Execute $BatPath -WorkingDirectory $ProjectRoot
        Trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At '05:30'
        Settings = New-ScheduledTaskSettingsSet @settingArgs
    }
}


function Show-WeeklyTuningPlan {
    # 설명: 참고 설정과 만들 작업의 핵심 값을 화면에 보여 준다.
    # 주의 사항: 출력만 한다.
    # 입력 타입->반환 타입: (PSCustomObject, hashtable) -> None
    param($Reference, $Parts)
    Write-Host "참고 작업 '$ReferenceTaskName':" ($Reference | Format-List | Out-String)
    Write-Host "만들 작업 '$WeeklyTaskName':"
    Write-Host "  실행: $BatPath (작업 폴더 $ProjectRoot)"
    Write-Host "  트리거: 매주 일요일 05:30"
    $s = $Parts.Settings
    Write-Host ("  WakeToRun={0} StartWhenAvailable={1} DisallowStartIfOnBatteries={2} StopIfGoingOnBatteries={3} " +
                "MultipleInstances={4} ExecutionTimeLimit={5}") -f $s.WakeToRun, $s.StartWhenAvailable,
                $s.DisallowStartIfOnBatteries, $s.StopIfGoingOnBatteries, $s.MultipleInstances, $s.ExecutionTimeLimit
}


function Register-WeeklyTuningTask {
    # 설명: Weekly Tuning 작업을 등록한다. 참고 작업이 저장된 암호 방식이면 같은 방식으로 자격 증명을 받아 등록한다.
    # 주의 사항: 같은 이름 작업이 있으면 멈춘다(덮어쓰지 않음). 사람이 -Register로 직접 실행할 때만 부른다.
    # 입력 타입->반환 타입: (PSCustomObject, hashtable) -> None
    param($Reference, $Parts)
    if (Get-ScheduledTask -TaskName $WeeklyTaskName -ErrorAction SilentlyContinue) {
        throw "작업 '$WeeklyTaskName'이 이미 있다 — 덮어쓰지 않는다. 지우거나 이름을 바꾼 뒤 다시 실행할 것."
    }
    $common = @{ TaskName = $WeeklyTaskName; Action = $Parts.Action; Trigger = $Parts.Trigger; Settings = $Parts.Settings }
    if ($Reference.LogonType -eq 'Password') {
        $cred = Get-Credential -UserName "$env:USERDOMAIN\$env:USERNAME" -Message "'$WeeklyTaskName' 작업을 실행할 계정 암호"
        Register-ScheduledTask @common -User $cred.UserName -Password $cred.GetNetworkCredential().Password | Out-Null
    } else {
        Register-ScheduledTask @common | Out-Null
    }
    Write-Host "등록됨: $WeeklyTaskName"
}


if (-not (Test-Path $BatPath)) { throw "bat 없음: $BatPath" }
$reference = Read-ReferenceTaskSettings
$parts = New-WeeklyTuningTaskParts -Reference $reference
Show-WeeklyTuningPlan -Reference $reference -Parts $parts
if ($Register) {
    Register-WeeklyTuningTask -Reference $reference -Parts $parts
} else {
    Write-Host "등록하지 않았다(보기만). 등록하려면 -Register를 붙여 다시 실행한다."
}
