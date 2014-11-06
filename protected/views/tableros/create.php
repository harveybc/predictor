<?php
    if (isset($_GET['id']))
    {
        $modeloId = Motores::model()->findByAttributes(array ("Area" => $_GET['id']));
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

    .selecta{

        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;


    }



</style> 

<script type="text/javascript">
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
        
        
      
<?php
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('proceso' => 'js:document.getElementById("Tableros_Proceso").value'),
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
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('area' => 'js:document.getElementById("Tableros_Area").value'),
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
       
    function updateFieldEquipoVacio()
    {
<?php
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('area' => 'js:document.getElementById("Tableros_Area").value'),
        'url' => CController::createUrl('/motores/dynamicEquipoVacio'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        'success' => 'updateEquipoDropdownVacio',
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
    
    function updateFieldEquipoComment()
    {
<?php
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('area' => 'js:document.getElementById("Tableros_Area").value'),
        'url' => CController::createUrl('/motores/dynamicEquipoVacio'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        'success' => 'updateAreaDropdownComment',
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

    function updateGridReportes()
    {
<?php
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('area' => 'js:document.getElementById("Tableros_Area").value', 'equipo' => 'js:document.getElementById("Tableros_Equipo").value'),
        'update' => '#divGridReportes',
        'url' => CController::createUrl('/reportes/dynamicReportes'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
    );
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    function updateGridReportesArea()
    {
<?php
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('area' => 'js:document.getElementById("Tableros_Area").value'),
        'update' => '#divGridReportes',
        'url' => CController::createUrl('/reportes/dynamicReportesArea'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
    );
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }


    // función que cambia los datos de el Dropdownlist de Area
    function updateAreaDropdown(data)
    {
        $('#Tableros_Area').html(data.value1);
        updateFieldEquipo();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#Tableros_Equipo').html(data.value1);
        //updateGridReportes();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdownVacio(data)
    {   
        updateFieldEquipoComment();
        updateGridReportesArea();
    };
    
    // función que coloca el dropdown de Equipo  con comentario únicamente
    function updateAreaDropdownComment(data)
    {
        $('#Tableros_Equipo').html(data.value1);
        //updateFieldEquipo();
    };
    


</script>


<?php
    $this->breadcrumbs = array (
        'Tableros' => array (Yii::t('app',
                                    'index')),
        Yii::t('app',
               'Crear'),
    );

    $this->menu = array (
        array ('label' => 'Lista de Tableros', 'url' => array ('index')),
        array ('label' => 'Gestionar Tableros', 'url' => array ('admin')),
    );
?>

<?php $this->setPageTitle (' Nuevo Tablero '); ?>
<div class="form">

    <?php
        $form = $this->beginWidget('CActiveForm',
                                   array (
                'id' => 'tableros-form',
                'enableAjaxValidation' => true,
            ));
    ?>
<div name="myDiv" id="myDiv" class="forms50cb">
    
        <div styler="width:100%;">
            <div>
                <div styler="width:50%;">
                    <b>Area:</b>
                    <?php
                        if (isset($modeloId))
                        {
                            $model->Proceso = $modeloId->Proceso;
                        }
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                        echo CHtml::activeDropDownList($model,
                                                       'Proceso',
                                                       CHtml::listData(Estructura::model()->findAllbySql(
                                    'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC',
                                    array ()),
                                    'Proceso',
                                    'Proceso'
                            ),
                                    array (
                            //'onfocus' => 'updateFieldArea()',
                            'onchange' => 'updateFieldArea()',
                            'style' => 'width:100%;',
                            'class' => 'select',
                            'empty' => 'Seleccione el área',
                            )
                        );
                    ?>
                    <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
                </div>
                <div styler="width:50%;">
                    <b>Proceso:</b>
                    <?php
                        if (isset($modeloId))
                        {
                            $model->Area = $modeloId->Area;
                        }
// dibuja el dropDownList de Area, dependiendo del proceso selecccionado
                        echo CHtml::activeDropDownList($model,
                                                       'Area',
                                                       CHtml::listData(Estructura::model()->findAllbySql(
                                    'SELECT DISTINCT Area FROM tableros WHERE Proceso="'.(isset($modeloId)?$modeloId->Proceso:"ELABORACION").'" ORDER BY Area ASC',
                                    array ()),
                                    'Area',
                                    'Area'
                            ),
                                    array (
                            'style' => 'width:100%;',
                            'class' => 'select',
                            'empty' => 'Seleccione el proceso',
                            )
                        );
                    ?>
                </div>
            </div>
            </div>
    </div>

<?php
    echo $this->renderPartial('_form',
                              array (
        'model' => $model,
        'form' => $form
    ));
?>
        

    

    
    <div class="row buttons forms100c">
<?php echo CHtml::submitButton(Yii::t('app',
                                                                                                         'Crear')); ?>
    </div>

<?php $this->endWidget(); ?>

</div>
