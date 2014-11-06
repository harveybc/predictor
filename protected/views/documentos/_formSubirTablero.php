<style type="text/css">

    .tableDoc{
        width:888px; 
        vertical-align:middle;

    }
     .table{
            width:888px; 
            vertical-align:middle;
           

        }

    .table2{
        width:888px; 

        vertical-align:middle;


    }

    .marcoDoc{

        width:888px !important;
        height:174px;margin-left:0px;
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
        height:347px;margin-left:0px;
        margin-bottom:5px !important;
        border-color:#961C1F;
        padding-top:5px;
        padding-right:5px;
        padding-left:14px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */

    }

    .marcoMeta2 {

        width:888px !important;
        height:90px;margin-left:0px;
        margin-bottom:5px !important;
        border-color:#961C1F;
        padding-top:5px;
        padding-right:5px;
        padding-left:14px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */

    }

    .secuenciaDoc{

        width: 200px;
        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;

    }

    .secuencias{

        width: 150px;
        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;

    }

    .select1{
        width:325px;
        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;


    }
     


</style>

<script>
    function cambiarTitulo(){
        
        var conv=$('#Documentos_descripcion').attr('value');
        $('#MetaDocs_titulo').attr('value',conv);
    }
</script>


<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary(array($model, $modelMeta,$modelArchivo)); ?>

<br/>
<br/>

<?php echo $form->errorSummary(array($model, $modelMeta,$modelArchivo)); ?>

<?php
$this->widget('zii.widgets.jui.CJuiAccordion', array(
    'panels' => array(
        'Datos Básicos' => $this->renderPartial('_accordionSubirTablero', array('form' => $form, 'modelMeta' => $modelMeta, 'modelArchivo' => $modelArchivo, 'modeloId'=>$modeloId, 'model' => $model, 'panel' => "0"), true),
        'Opciones Avanzadas' => $this->renderPartial('_accordionSubirTablero', array('form' => $form, 'modelMeta' => $modelMeta, 'model' => $model, 'panel' => "1"), true),
        'Datos de Creación y Revisión' => $this->renderPartial('_accordionSubirTablero', array('form' => $form, 'modelMeta' => $modelMeta, 'model' => $model, 'panel' => "2"), true),
    ),
    // additional javascript options for the accordion plugin
    'options' => array(
        'animated' => 'bounceslide',
    ),
    'htmlOptions' => array('style' => 'width:920px;margin:0px;padding:0px;'),
    'themeUrl' => '/themes',
    'theme' => 'acordeonSgdoc',
));
?>
