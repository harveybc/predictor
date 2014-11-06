
<style>

    .tableDoc{
        width:888px; 
        vertical-align:middle;

    }

    .tableDoc2{
        width:850px;
        padding-left:30px

    }


    table{
        width:888px; 
        vertical-align:middle;
        margin:0px;
        padding:0px;

    }

    .table2{
        width:888px; 

        vertical-align:middle;


    }

    .marcoDoc{

        width:888px !important;
        
        margin-bottom:5px !important;
        border-color:#961C1F;
        padding-top:5px;
        padding-right:5px;
        padding-left:14px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */
    }

    .marcoMeta {

        width:888px !important;
    
        margin:0px !important;
        border-color:#961C1F;
        padding:0px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */

    }

    .marcoMeta2 {

        width:888px !important;
   
        margin:0px !important;
        border-color:#961C1F;
        padding:0px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */

    }

    .secuenciaDoc{

        width: 200px;

    }

    .secuencias{

        width: 150px;

    }
    
    .secuencias2{

        width: 303px;

    }
    
    .select1{
        margin-right:13px;
        width:360px;
        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:3px;
    }
    td{

        padding:0px;
        margin:0px;

    }


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

    .select{

        width: 30px;
        padding-left:2px;
        background:#ffffff;
        background-color:#ffffff;
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 3px;
        border-radius:3px;


    }
    .back{

               
       padding-left:605px


    }




</style>



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

    .select{

        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;


    }



</style>


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

    .select{

        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;


    }



</style>


<script type="text/javascript">
    
    function updateGridEstructura()
    {
        
        
      
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("Estructura_Area").value'),
    'url' => CController::createUrl('/estructura/dynamicEstructura'), //url to call.
    'update' => '#divEstructura', //selector to update
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
        
        
      
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('proceso' => 'js:document.getElementById("MetaDocs_UT_Area").value'),
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
    'data' => array('area' => 'js:document.getElementById("MetaDocs_UT_Proceso").value'),
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
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
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
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
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
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value', 'equipo' => 'js:document.getElementById("equipo").value'),
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
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
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
        $('#MetaDocs_UT_Proceso').html(data.value1);
        //  $("#area").trigger("liszt:updated");
        // $("#equipo").trigger("liszt:updated");
        //updateGridEstructura();
        //updateFieldEquipo();
        //updateGridTableros();
        
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#MetaDocs_UT_Equipo').html(data.value1);
        //updateGridReportes();
        //  $("#equipo").trigger("liszt:updated");
        // $("#motor").trigger("liszt:updated");
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdownVacio(data)
    {   
        updateFieldEquipoComment();
        updateGridReportesArea();
        //  $("#equipo").trigger("liszt:updated");
    };
    
    // función que coloca el dropdown de Equipo  con comentario únicamente
    function updateAreaDropdownComment(data)
    {
        $('#equipo').html(data.value1);
        //   $("#motor").trigger("liszt:updated");
        //updateFieldEquipo();
    };
    


</script>

<script>
    function cambiarTitulo(){
        
        var conv=$('#Documentos_descripcion').attr('value');
        $('#MetaDocs_titulo').attr('value',conv);
    }
</script>


<?php
if (0 + $panel == 0) {
    ?>

    <table>
        <tr>
            <td style="padding-left:10px;">           
                <div class="row">
                    <?php echo $form->labelEx($model, 'descripcion'); ?>
                    <?php echo $form->textField($model, 'descripcion', array('size' => 60, 'maxlength' => 128, 'style' => 'width:300px;margin-right:13px;', 'onchange' => 'cambiarTitulo()')); ?>
                    <?php echo $form->error($model, 'descripcion'); ?>
                </div>
            </td>
            <td style="width:270px">

                <?php
                echo $form->labelEx($modelArchivo, 'nombre');
                echo $form->fileField($modelArchivo, 'nombre', array( 'style' => 'width:200px;'));
                echo $form->error($modelArchivo, 'nombre');
                ?>
            </td>
            <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'autores'); ?>
                    <?php echo $form->textField($modelMeta, 'autores', array('size' => 60, 'maxlength' => 256, 'style' => 'width:200px')); ?>
                    <?php echo $form->error($modelMeta, 'autores'); ?>
                </div>
            </td>
        </tr>
         </table>      

        <table style="padding-left:10px;width:900px;margin-left:0px;margin-bottom:5px !important; ">
            <tr>
                <td >
                    <b>Area:</b>
                    <?php
                    $valor = isset($modeloId) ? $modeloId->Proceso : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                    echo CHtml::dropDownList(
                            'MetaDocs[UT_Area]', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                            'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
                            ), array(
                        //'onfocus' => 'updateFieldArea()',
                        'onchange' => 'updateFieldArea();',
                        'style' => 'width:170px;margin-right:15px',
                        'class' => 'select'
                            )
                    );
                    ?>
                    <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
                </td>
                <td>
                    <b>Proceso:</b>
                    <?php
                    $valor = isset($modeloId) ? $modeloId->Area : "";
// dibuja el dropDownList de Area, dependiendo del proceso selecccionado
                    echo CHtml::dropDownList(
                            'MetaDocs[UT_Proceso]', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                            isset($modeloId) ?
                                                    'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC' : 'SELECT DISTINCT Area FROM estructura WHERE Proceso="ELABORACION" ORDER BY Area ASC', array()), 'Area', 'Area'
                            ), array(
                        'onchange' => 'updateFieldEquipo();',
                        'style' => 'width:170px;margin-right:15px',
                        'class' => 'select',
                        'empty' => 'Seleccione el proceso',
                            )
                    );
                    ?>
                </td>
        
                <td>
                    <b>Equipo:<br/></b>
                    <?php
                        if (isset($modeloId))
                            {
                            $model->Equipo = $modeloId->Equipo;
                            }
                        // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                        
                        echo CHtml::activeDropDownList($modelMeta,
                                                       'UT_Equipo',
                                                       CHtml::listData(Estructura::model()->findAllbySql(
                                    'SELECT Equipo FROM estructura WHERE Area="' .(isset($modeloId)? $modeloId->Area:"FILTRACION" ). '" ORDER BY Equipo ASC',
                                                                                                    array ()),
                                                                                                    'Equipo',
                                                                                                    'Equipo'
                            ),
                                                       array (
                            //'onfocus'  => 'updateFieldEquipo()',
                            //'onchange' => 'updateGridReportes()',
                            'style' => 'width:490px;margin-right:40px',
                            'class' => 'select',
                            'empty' => 'Seleccione el equipo',
                            )
                        );
                    ?>
                </td>
            </tr>
        </table> 

    <?php
}

if (0 + $panel == 1) {
    ?>

    <table class="table">
        <tr>


            <td>
                <div class="row">

    <?php echo $form->hiddenField($modelMeta, 'titulo', array('size' => 60, 'maxlength' => 256, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'titulo'); ?>
                </div>
                <div class="row">
    <?php echo $form->labelEx($modelMeta, 'descripcion'); ?>
                    <?php echo $form->textArea($modelMeta, 'descripcion', array('size' => 60, 'maxlength' => 256, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'descripcion'); ?>
                </div>

            </td>
            <td>
    <?php echo $form->labelEx($modelMeta, 'fechaRecepcion'); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($modelMeta->fechaRecepcion))
                    $today = $modelMeta->fechaRecepcion;
                else
                    $modelMeta->fechaRecepcion = $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $modelMeta, //Model object
                    'attribute' => 'fechaRecepcion', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendariocbm',
                    'htmlOptions' => array('style' => 'width:140px;'),
                    'options' => array(
                        'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                        'buttonImage' => '/images/calendar.png',
                        'showOn' => "both",
                        'buttonText' => "Seleccione la fecha",
                        'buttonImageOnly' => true,
                    ) // jquery plugin options
                ));
                ?>
            </td> 


            <td>
                <div class="row">
    <?php echo $form->labelEx($modelMeta, 'ISBN'); ?>
                    <?php echo $form->textField($modelMeta, 'ISBN', array('size' => 32, 'maxlength' => 32, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'ISBN'); ?>
                </div>
            </td>

            <td>
                <div class="row">
    <?php echo $form->labelEx($modelMeta, 'numPedido'); ?>
                    <?php echo $form->textField($modelMeta, 'numPedido', array('size' => 60, 'maxlength' => 64, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'numPedido'); ?>
                </div>
            </td>


        </tr>

        <tr>
            <td>
                <div class="row">
    <?php echo $form->labelEx($modelMeta, 'numComision'); ?>
                    <?php echo $form->textField($modelMeta, 'numComision', array('size' => 60, 'maxlength' => 64, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'numComision'); ?>
                </div>                
            </td>
            <td>
                <div class="row">
    <?php echo $form->labelEx($modelMeta, 'version'); ?>
                    <?php echo $form->textField($modelMeta, 'version', array('size' => 60, 'maxlength' => 64, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'version'); ?>
                </div>
            </td>

            <td>
                <label for="Cervecerias">Cerveceria</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $modelMeta,
                    'relation' => 'cerveceria0',
                    'fields' => 'descripcion',
                    'allowEmpty' => true,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuencias',)
                        )
                );
                    ?>

            </td>
            <td>
                <label for="Fabricantes">Fabricante</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $modelMeta,
                'relation' => 'fabricante0',
                'fields' => 'descripcion',
                'allowEmpty' => false,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                    'class' => 'secuencias',)
                    )
            );
                    ?>
            </td>


        </tr>
        <tr>
            <td>
                <label for="Idiomas">Idioma</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $modelMeta,
                'relation' => 'idioma0',
                'fields' => 'descripcion',
                'allowEmpty' => false,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                    'class' => 'secuencias',)
                    )
            );
                    ?>



            </td>
            <td>
    <?php echo $form->hiddenField($modelMeta, 'documento', array('value' => 1)); ?>

                <label for="TipoContenidos">Tipo de Contenido</label><?php
    $this->widget('application.components.Relation', array(
        'model' => $modelMeta,
        'relation' => 'tipoContenido0',
        'fields' => 'descripcion',
        'allowEmpty' => false,
        'style' => 'dropdownlist',
        'htmlOptions' => array(
            'class' => 'secuencias',)
            )
    );
    ?>

            </td>

            <td>
                <label for="Medios">Medio</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $modelMeta,
                'relation' => 'medio0',
                'fields' => 'descripcion',
                'allowEmpty' => false,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                    'class' => 'secuencias',)
                    )
            );
    ?>

            </td>

            <td>
    <?php echo $form->labelEx($model, 'conservacionInicio'); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($model->conservacionInicio))
                    $today = $model->conservacionInicio;
                else
                    $model->conservacionInicio = $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $model, //Model object
                    'attribute' => 'conservacionInicio', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendariocbm',
                    'htmlOptions' => array('style' => 'width:140px;'),
                    'options' => array(
                        'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                        'buttonImage' => '/images/calendar.png',
                        'showOn' => "both",
                        'buttonText' => "Seleccione la fecha",
                        'buttonImageOnly' => true,
                    ) // jquery plugin options
                ));
                ?>
            </td>
        </tr>
        <tr>
            <td>
    <?php echo $form->labelEx($model, 'conservacionFin'); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($model->conservacionFin))
                    $today = $model->conservacionFin;
                //    else
                // $model->conservacionFin = $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $model, //Model object
                    'attribute' => 'conservacionFin', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendariocbm',
                    'htmlOptions' => array('style' => 'width:140px;'),
                    'options' => array(
                        'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                        'buttonImage' => '/images/calendar.png',
                        'showOn' => "both",
                        'buttonText' => "Seleccione la fecha",
                        'buttonImageOnly' => true,
                    ) // jquery plugin options
                ));
                ?>
            </td>
            <td>
                <label for="Secuencias">Secuencia</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $model,
                'relation' => 'secuencia0',
                'fields' => 'descripcion',
                'allowEmpty' => false,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                    'class' => 'secuenciaDoc',)
                    )
            );
                ?>

            </td>
            <td>
                <label for="OrdenSecuencias">Orden de secuencia</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $model,
                'relation' => 'ordenSecuencia0',
                'fields' => 'posicion',
                'allowEmpty' => true,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                    'class' => 'secuenciaDoc',)
                    )
            );
                ?>
            </td>
            <td>
                <div class="row">
    <?php echo $form->labelEx($model, 'conservacionPermanente'); ?>
                    <?php echo $form->checkBox($model, 'conservacionPermanente'); ?>
                    <?php echo $form->error($model, 'conservacionPermanente'); ?>
                </div>
            </td>


        </tr>

        <tr>
            <td>

                <div class="row"> 
    <?php echo $form->labelEx($model, 'permitirAdiciones'); ?>
                    <?php echo $form->checkBox($model, 'permitirAdiciones'); ?>
                    <?php echo $form->error($model, 'permitirAdiciones'); ?>
                </div>
            </td>
            <td>
                <div class="row">
    <?php echo $form->labelEx($model, 'permitirAnotaciones'); ?>
                    <?php echo $form->checkBox($model, 'permitirAnotaciones'); ?>
                    <?php echo $form->error($model, 'permitirAnotaciones'); ?>
                </div>
            </td>
            <td>
                <div class="row">
    <?php echo $form->labelEx($model, 'autorizarOtros'); ?>
                    <?php echo $form->checkBox($model, 'autorizarOtros'); ?>
                    <?php echo $form->error($model, 'autorizarOtros'); ?>
                </div>
            </td>
            <td>
                <div class="row">
    <?php echo $form->labelEx($model, 'requiereAutorizacion'); ?>
                    <?php echo $form->checkBox($model, 'requiereAutorizacion'); ?>
                    <?php echo $form->error($model, 'requiereAutorizacion'); ?>
                </div>
            </td>
        </tr>
        <tr>
            <td>


            </td>
        </tr>



    </table>


    <?php
}

if (0 + $panel == 2) {
    ?>


    <p class="note">Campos con<span class="required">*</span> son necesarios.</p>

    <?php echo $form->errorSummary(array($model, $modelMeta)); ?>


    <table>
        <tr>
            <td>

    <?php echo $form->labelEx($modelMeta, 'fechaCreacion'); ?>
    <?php
//Fecha inicial
    $today = date("Y-m-d H:i:s");
    if (isset($modelMeta->fechaCreacion))
        $today = $modelMeta->fechaCreacion;
    else
        $modelMeta->fechaCreacion = $today;
//fin Fecha inicial
    Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
    $this->widget('CJuiDateTimePicker', array(
        'model' => $modelMeta, //Model object
        'attribute' => 'fechaCreacion', //attribute name
        'mode' => 'datetime', //use "time","date" or "datetime" (default)
        'language' => 'es',
        //    'value' =>$today,
        'themeUrl' => '/themes',
        'theme' => 'calendariocbm',
        'htmlOptions' => array('style' => 'width:140px;'),
        'options' => array(
            'dateFormat' => 'yy-mm-dd',
            'showButtonPanel' => true,
            "yearRange" => '1995:2070',
            'changeYear' => true,
            'buttonImage' => '/images/calendar.png',
            'showOn' => "both",
            'buttonText' => "Seleccione la fecha",
            'buttonImageOnly' => true,
        ) // jquery plugin options
    ));
    ?>
            </td>
            <td>
                <label for="Usuarios">Documento subido por Usuario</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $modelMeta,
                'relation' => 'usuario0',
                'fields' => 'Username',
                'allowEmpty' => false,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                    'class' => 'secuencias',)
                    )
            );
    ?>
            </td>

            <td>
    <?php echo $form->labelEx($modelMeta, 'fechaRevisado'); ?>
    <?php
//Fecha inicial
    $today = date("Y-m-d H:i:s");
    if (isset($modelMeta->fechaRevisado))
        $today = $modelMeta->fechaRevisado;
    //   else
    //   $modelMeta->fechaRevisado= $today;
//fin Fecha inicial
    Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
    $this->widget('CJuiDateTimePicker', array(
        'model' => $modelMeta, //Model object
        'attribute' => 'fechaRevisado', //attribute name
        'mode' => 'datetime', //use "time","date" or "datetime" (default)
        'language' => 'es',
        //    'value' =>$today,
        'themeUrl' => '/themes',
        'theme' => 'calendariocbm',
        'htmlOptions' => array('style' => 'width:140px;'),
        'options' => array(
            'dateFormat' => 'yy-mm-dd',
            'showButtonPanel' => true,
            "yearRange" => '1995:2070',
            'changeYear' => true,
            'buttonImage' => '/images/calendar.png',
            'showOn' => "both",
            'buttonText' => "Seleccione la fecha",
            'buttonImageOnly' => true,
        ) // jquery plugin options
    ));
    ?>
            </td>


            <td>
                <label for="Usuarios">Fué revisado por</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $modelMeta,
                'relation' => 'userRevisado0',
                'fields' => 'Username',
                'allowEmpty' => false,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                    'class' => 'secuencias',)
                    )
            );
    ?>

            </td>

        </tr>


    </table>




    <?php
}
?>

