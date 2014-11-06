<style type="text/css">



   
.secuencias{

        width: 150px;
        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;

    }

.select2{
        width:300px;
        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;


    }


</style>

<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

<div styler="width:650px !important;height:438px;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:10px;padding-right:5px;padding-left:14px; ">

    <div>
        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'numPedido'); ?>
                    <?php echo $form->textField($model, 'numPedido', array('size' => 60, 'maxlength' => 64, 'style' => 'width:100px')); ?>
                    <?php echo $form->error($model, 'numPedido'); ?>
                </div>
            </div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'numComision'); ?>
                    <?php echo $form->textField($model, 'numComision', array('size' => 60, 'maxlength' => 64, 'style' => 'width:100px')); ?>
                    <?php echo $form->error($model, 'numComision'); ?>
                </div>                
            </div>

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'titulo'); ?>
                    <?php echo $form->textField($model, 'titulo', array('size' => 60, 'maxlength' => 256)); ?>
                    <?php echo $form->error($model, 'titulo'); ?>
                </div>


            </div>
        </div>
    </div>
    <div>
        <div>
            <div class="row">
                <?php echo $form->labelEx($model, 'descripcion'); ?>
                <?php echo $form->textArea($model, 'descripcion', array('size' => 60, 'maxlength' => 256, 'style' => 'width:620px')); ?>
                <?php echo $form->error($model, 'descripcion'); ?>
            </div>

        </div>
        </div>
    </div>

    <div>
        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'version'); ?>
                    <?php echo $form->textField($model, 'version', array('size' => 60, 'maxlength' => 64, 'style' => 'width:100px')); ?>
                    <?php echo $form->error($model, 'version'); ?>
                </div>
            </div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'disponibles'); ?>
                    <?php echo $form->textField($model, 'disponibles', array('style' => 'width:60px')); ?>
                    <?php echo $form->error($model, 'disponibles'); ?>
                </div>
            </div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'existencias'); ?>
                    <?php echo $form->textField($model, 'existencias', array('style' => 'width:60px')); ?>
                    <?php echo $form->error($model, 'existencias'); ?>
                </div>
            </div>

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'modulo'); ?>
                    <?php echo $form->textField($model, 'modulo', array('style' => 'width:60px')); ?>
                    <?php echo $form->error($model, 'modulo'); ?>
                </div>
            </div>
            <div>

                <div class="row">
                    <?php echo $form->labelEx($model, 'columna'); ?>
                    <?php echo $form->textField($model, 'columna', array('style' => 'width:60px')); ?>
                    <?php echo $form->error($model, 'columna'); ?>
                </div>

            </div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'fila'); ?>
                    <?php echo $form->textField($model, 'fila', array('style' => 'width:60px')); ?>
                    <?php echo $form->error($model, 'fila'); ?>
                </div>
            </div>
        </div>
    </div>
    <div>
        <div>

            <div>
                <?php echo $form->labelEx($model, 'fechaCreacion'); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($model->fechaCreacion))
                    $today = $model->fechaCreacion;
             //   else
                  //  $model->fechaCreacion= $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $model, //Model object
                    'attribute' => 'fechaCreacion', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendarioCbm',
                    'htmlOptions'=>array('style'=>'width:140px;'),
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
            </div>
                
        </div>
    </div>
    <div>
        <div>
 <div>
              
                <?php echo $form->labelEx($model, 'fechaRecepcion'); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($model->fechaRecepcion))
                    $today = $model->fechaRecepcion;
               else
                    $model->fechaCreacion= $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $model, //Model object
                    'attribute' => 'fechaRecepcion', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendarioCbm',
                    'htmlOptions'=>array('style'=>'width:140px;'),
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
            </div> 

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'revisado'); ?>
                    <?php echo $form->checkBox($model, 'revisado'); ?>
<?php echo $form->error($model, 'revisado'); ?>
                </div>


            </div>

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'fechaRevisado'); ?>
                    <?php echo $form->textField($model, 'fechaRevisado'); ?>
<?php echo $form->error($model, 'fechaRevisado'); ?>
                </div>
            </div>

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'autores'); ?>
                    <?php echo $form->textField($model, 'autores', array('size' => 60, 'maxlength' => 256,'style'=>'width:200px')); ?>
<?php echo $form->error($model, 'autores'); ?>
                </div>

            </div>
            <div>

            </div>
        </div>
    </div>
</div>
<div styler="width:650px!important;height:280px;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:12px;padding-right:5px;padding-left:14px; ">

    <div>
        <div>
            <div>
                <label for="TipoContenidos">Tipo de Contenido</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'tipoContenido0',
    'fields' => 'descripcion',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>

            </div>
            <div>
                <label for="Fabricantes">Fabricantes</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'fabricante0',
    'fields' => 'descripcion',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>
            </div>

            <div>
                <label for="Cervecerias">Cervecerias </label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'cerveceria0',
    'fields' => 'descripcion',
    'allowEmpty' => true,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>

            </div>
        </div>
        <div>
            <div>
                <label for="UbicacionTec">Ubicación Técnica</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'ubicacionT0',
    'fields' => 'descripcion',
    'allowEmpty' => true,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>
            </div>

            <div>
                <label for="Medios">Medio</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'medio0',
    'fields' => 'descripcion',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>

            </div>
            <div>
                <label for="Idiomas">Idioma</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'idioma0',
    'fields' => 'descripcion',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>
            </div>
        </div>
        <div>
            <div>
                <label for="Documentos">Unidad documental</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'documento0',
    'fields' => 'descripcion',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>

            </div>
            <div>
                <label for="Usuarios">Registrado por</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'usuario0',
    'fields' => 'nombre',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>
            </div>

            <div>
                <label for="Usuarios">Fué revisado por</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'userRevisado0',
    'fields' => 'nombre',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>

            </div>
        </div>
        <div>
            <div>
                <label for="Secuencias">Pertenece a la siguiente secuencia</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'secuencia0',
    'fields' => 'descripcion',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>
            </div>



            <div>
                <label for="OrdenSecuencias">Orden dentro de secuencia</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'ordenSecuencia0',
    'fields' => 'posicion',
    'allowEmpty' => true,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
        'class' => 'secuencias',)
        )
);
?>
            </div>
        </div>

        <div>

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'ISBN'); ?>
                    <?php echo $form->textField($model, 'ISBN', array('size' => 32, 'maxlength' => 32, 'style' => 'width:100px')); ?>
<?php echo $form->error($model, 'ISBN'); ?>
                </div>
            </div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'EAN13'); ?>
                    <?php echo $form->textField($model, 'EAN13', array('size' => 32, 'maxlength' => 32, 'style' => 'width:100px')); ?>
<?php echo $form->error($model, 'EAN13'); ?>
                </div>
            </div>


        </div>
    </div>

</div>






































