<?php
if (isset($_GET['id'])) {
    $modeloId = Motores::model()->findByAttributes(array("TAG" => $_GET['id']));
}
?>
<style type="text/css">
    div.loading {
        background-color: #FFFFFF;
        background-image: url('/images/loading.gif');
        background-position:  100px;
        background-repeat: no-repeat;
        opacity: 1;
    }
    div.loading * {
        opacity: .8;
    }
</style>
<script type="text/javascript">
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
                
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('proceso' => 'js:document.getElementById("proceso").value'),
    'url' => CController::createUrl('/motores/dynamicArea'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateAreaDropdown',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;

        return false;
    }
    
    function updateFieldEquipo()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/motores/dynamicEquipo'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateEquipoDropdown',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    function updateFieldMotor()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('equipo' => 'js:document.getElementById("equipo").value'),
    'url' => CController::createUrl('/motores/dynamicFMotor'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateMotorDropdown',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    //actualiza el divGraph
    // function updateGraph()
    {



        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        //  return false;
    }

    function updateGridFechas()
    {
        $('#linkNuevo').html('<a href="/index.php/aislamiento_tierra/create?id='+document.getElementById("motor").value+'">Nueva medición</a>');
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('TAG' => 'js:document.getElementById("motor").value'),
    'update' => '#gridFechas',
    'url' => CController::createUrl('/aceitesnivel1/dynamicFechas'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        //  updateGraph();
        return false;
    }

    // función que cambia los datos de el Dropdownlist de Area
    

    // función que cambia los datos de el Dropdownlist de Area
    function updateAreaDropdown(data)
    {
        $('#area').html(data.value1);
        $('#equipo').html('<option value="0"></option>');
        $('#motor').html('<option value="0"></option>');
        // actualiza el echosen de area
        $("#area").trigger("liszt:updated");
        $("#equipo").trigger("liszt:updated");
        $("#motor").trigger("liszt:updated");
        updateFieldEquipo();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#equipo').html(data.value1);
        $('#motor').html('<option value="0"></option>');
        $("#equipo").trigger("liszt:updated");
        $("#motor").trigger("liszt:updated");
        updateFieldMotor();
    };
    
    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdownVacio(data)
    {   
        updateFieldEquipoComment();
        updateGridReportesArea();
        $("#equipo").trigger("liszt:updated");
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateMotorDropdown(data)
    {
        $('#Aceitesnivel1_TAG').html(data.value1);
        
        //updateGridFechas();
        //updateGraph();
    };


</script>

<?php
$this->breadcrumbs = array(
    'Lubricantes' => array(Yii::t('app', 'index')),
    Yii::t('app', 'Crear'),
);

$this->menu = array(
    array('label' => 'Lista de Mediciones', 'url' => array('index')),
    array('label' => 'Gestionar Mediciones', 'url' => array('admin')),
);
?>

<?php $this->setPageTitle(' Crear Medición de Lubricantes '); ?>


<div class="form">

    <?php
    $form = $this->beginWidget('CActiveForm', array(
        'id' => 'aceitesnivel1-form',
        'enableAjaxValidation' => true,
            ));
    ?>
    <div name="myDiv" id="myDiv" class="forms100cb">
        <div styler="width:100%; ">
            <div>
                <div class="forms50c" styler="width:50%;">
                    <b>Area:</b>
                    <?php
                    // si existe el parámetro Id, configura los preseleccionados
                    $valor = isset($modeloId) ? $modeloId->Proceso : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                    echo CHtml::dropDownList(
                            'proceso', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                            'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
                            ), array(
                        //'onfocus' => 'updateFieldArea()',
                        'onchange' => 'updateFieldArea()',
                        'styler' => 'width:100%;',
                        'class' => 'select',
                            )
                    );
                    ?>
                    <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
                </div>
                <div class="forms50c" styler="width:50%;">
                    <b>Proceso:</b>
                    <?php
                    $valor = isset($modeloId) ? $modeloId->Area : "";
                    // dibuja el dropDownList de Area, dependiendo del proceso selecccionado
                    echo CHtml::dropDownList(
                            'area', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                            isset($modeloId) ?
                                                    'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC' : 'SELECT DISTINCT Area FROM estructura WHERE Proceso="ELABORACION" ORDER BY Area ASC', array()), 'Area', 'Area'
                            ), array(
                        //'onfocus' => 'updateFieldArea()',
                        'onchange' => 'updateFieldEquipo()',
                        'styler' => 'width:100%;',
                        'class' => 'select',
                        'empty' => 'Seleccione el proceso',
                            )
                    );
                    ?>
                </div>
            </div>
        </div>


        <div styler="width:100%; ">
            <div>
                <div>
                    <b>Equipo:</b>
                    <?php
                    $valor = isset($modeloId) ? $modeloId->Equipo : "";
                    // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                    echo CHtml::dropDownList(
                            'equipo', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                            isset($modeloId) ?
                                                    'SELECT Equipo FROM estructura WHERE Area="' . $modeloId->Area . '" ORDER BY Equipo ASC' : 'SELECT Equipo FROM estructura WHERE Area="FILTRACION" ORDER BY Equipo ASC', array()), 'Equipo', 'Equipo'
                            ), array(
                        //'onfocus' => 'updateFieldEquipo()',
                        'onchange' => 'updateFieldMotor()',
                        'styler' => 'width:100%;',
                        'class' => 'select',
                        'empty' => 'Seleccione el equipo',
                            )
                    );
                    ?>
                </div>
            </div>
            <div>
                <div>
                    <b>Motor:</b>
                    <?php
                    $valor = isset($modeloId) ? $modeloId->TAG : "";
                    // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                    echo CHtml::dropDownList(
                            'Aceitesnivel1[TAG]', $valor, CHtml::listData(Motores::model()->findAllbySql(
                                            isset($modeloId) ?
                                                    'SELECT TAG, CONCAT(TAG," - ",Motor) as Motor FROM motores WHERE Equipo="' . $modeloId->Equipo . '" ORDER BY TAG ASC' : 'SELECT TAG, CONCAT(TAG," - ",Motor) as Motor FROM motores WHERE Equipo="ANILLO DE CONTRAPRESION" ORDER BY TAG ASC', array()), 'TAG', 'Motor'
                            ), array(
                        //'onfocus' => 'updateGraph()',
                        // 'onchange' => 'updateGridFechas()',
                        'styler' => 'width:100%;',
                        'class' => 'select',
                        'empty' => 'Seleccione el motor para filtar el resultado',
                            )
                    );
                    ?>
                </div>
            </div>
        </div>
    </div>



<?php
echo $this->renderPartial('_form', array(
    'model' => $model,
    'form' => $form
));
?>

    <div class="row buttons forms100c">
                    <?php echo CHtml::submitButton(Yii::t('app', 'Guardar')); ?>
    </div>

                    <?php $this->endWidget(); ?>

</div>
